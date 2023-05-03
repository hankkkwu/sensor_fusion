"""
Early fusion - cameara and lidar
After loading data from the dataset, our Early fusion process will happen in 3 steps:
1.   **Project the Point Clouds (3D) to the Image(2D)** 
2.   **Detect Obstacles in 2D** (Camera)
3.   **Fuse the Results**
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import open3d as o3d

import struct

import statistics
import random

from yolov4.tf import YOLOv4
import tensorflow as tf
import time


# helper function - convert BIN TO PCD
def bin_to_pcd(point_files, index):
    """turn LiDAR file from binary extension '.bin' into a '.pcd' and save it"""
    size_float = 4
    list_pcd = []

    file_to_open = point_files[index]
    file_to_save = str(point_files[index])[:-3]+"pcd"
    with open (file_to_open, "rb") as f:
        byte = f.read(size_float*4)
        while byte:
            x,y,z,intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float*4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)

    o3d.io.write_point_cloud(file_to_save, pcd)


def run_obstacle_detection(img, yolo):
    start_time=time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = yolo.resize_image(img)
    # 0 ~ 255 to 0.0 ~ 1.0
    resized_image = resized_image / 255.
    #input_data == Dim(1, input_size, input_size, channels)
    input_data = resized_image[np.newaxis, ...].astype(np.float32)

    candidates = yolo.model.predict(input_data)

    _candidates = []
    result = img.copy()
    for candidate in candidates:
        batch_size = candidate.shape[0]
        grid_size = candidate.shape[1]
        _candidates.append(tf.reshape(candidate, shape=(1, grid_size * grid_size * 3, -1)))
        #candidates == Dim(batch, candidates, (bbox))
        candidates = np.concatenate(_candidates, axis=1)
        #pred_bboxes == Dim(candidates, (x, y, w, h, class_id, prob))
        pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0], iou_threshold=0.35, score_threshold=0.40)
        pred_bboxes = pred_bboxes[~(pred_bboxes==0).all(1)] #https://stackoverflow.com/questions/35673095/python-how-to-eliminate-all-the-zero-rows-from-a-matrix-in-numpy?lq=1
        pred_bboxes = yolo.fit_pred_bboxes_to_original(pred_bboxes, img.shape)
        exec_time = time.time() - start_time
        #print("time: {:.2f} ms".format(exec_time * 1000))
        result = yolo.draw_bboxes(img, pred_bboxes)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result, pred_bboxes


class LiDAR2Camera(object):
    """Project the Points in the Image"""
    def __init__(self, calib_file):
        #############################
        # Read the Calibration File #
        #############################
        calibs = self.read_calib_file(calib_file)
        # camera calibration matrix
        self.P = np.array(calibs['P2']).reshape(3,4)
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = np.array(calibs['Tr_velo_to_cam']).reshape(3,4)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = np.array(calibs['R0_rect']).reshape(3,3)

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    
    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom
    
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return pts_3d_velo @ self.V2C.T
    
    def project_velo_to_image(self, pts_3d_velo):
        """
        Input: 3D points in Velodyne Frame [nx3]
        Output: 2D Pixels in Image Frame [nx2]
        """
        R0_homo = np.concatenate([self.R0, np.zeros((3,1))], axis=1)
        R0_homo = np.concatenate([R0_homo, np.array([[0,0,0,1]])], axis=0)  # 4x4

        V2C_homo = np.concatenate([self.V2C, np.array([[0,0,0,1]])], axis=0)     # 4x4

        pts_3d_velo_homo = np.concatenate([pts_3d_velo, np.ones((pts_3d_velo.shape[0],1))], axis=1).T   # 4 x num_pts

        image_pixels = self.P @ R0_homo @ V2C_homo @ pts_3d_velo_homo
        image_pixels /= image_pixels[2,:]
        image_pixels = image_pixels[:2, :].T
        return image_pixels
    
    def get_lidar_in_image_fov(self, pc_velo, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0):
        """ Filter lidar points, keep those in image FOV """
        pts_2d = self.project_velo_to_image(pc_velo)
        # TODO: Remove pixels that are out of image boundaries
        fov_inds = (pts_2d[:, 0] >= xmin) & (pts_2d[:, 0] < xmax) & (pts_2d[:, 1] >= ymin) & (pts_2d[:, 1] < ymax)
        # TODO: Remove points that are closer than the clip distance
        fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
        # imgfov_pc_velo contains point coulds that with in the image boundaries
        imgfov_pc_velo = pc_velo[fov_inds, :]
        if return_more:
            return imgfov_pc_velo, pts_2d, fov_inds
        else:
            return imgfov_pc_velo
        
    def show_lidar_on_image(self, pc_velo, img, debug="False"):
        """ Project LiDAR points to image """
        imgfov_pc_velo, pts_2d, fov_inds = self.get_lidar_in_image_fov(
            pc_velo, 0, 0, img.shape[1], img.shape[0], True
        )
        if (debug==True):
            print("3D PC Velo "+ str(imgfov_pc_velo)) # The 3D point Cloud Coordinates
            print("2D PIXEL: " + str(pts_2d)) # The 2D Pixels
            print("FOV : "+str(fov_inds)) # Whether the Pixel is in the image or not
        
        imgfov_pts_2d = pts_2d[fov_inds, :]

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        
        for i in range(imgfov_pts_2d.shape[0]):
            # TODO: Draw a circle at the pixel position and depending on its color
            depth = imgfov_pc_velo[i, 0]

            # since the point clouds distance start from 2m, so 510/2 = 255
            color = cmap[int(510 / depth), :]

            x = int(np.round(imgfov_pts_2d[i, 0]))
            y = int(np.round(imgfov_pts_2d[i, 1]))
            cv2.circle(img, (x, y), 2, color=tuple(color), thickness=1)

        return img
    
    def lidar_camera_fusion(self, pred_bboxes, image, point_cloud):
        """implements the fusion between boxes and points"""
        img = image.copy()

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        imgfov_pc_velo, pts_2d, fov_inds = self.get_lidar_in_image_fov(point_cloud, 0, 0, img.shape[1], img.shape[0], True)
        imgfov_pts_2d = pts_2d[fov_inds, :]

        for bbox in pred_bboxes:
            x, y, w, h = bbox[:4]
            x = int(x * image.shape[1])
            y = int(y * image.shape[0])
            w = int(w * image.shape[1])
            h = int(h * image.shape[0])
            mask = rectContains(imgfov_pts_2d, x, y, w, h, shrink_factor=0.2)
            bbox_pts_2d = imgfov_pts_2d[mask]
            bbox_pts_3d = imgfov_pc_velo[mask, :]
            
            if len(bbox_pts_2d) <= 3:
                continue

            for i in range(len(bbox_pts_2d)):
                # TODO: Draw a circle at the pixel position and depending on its color
                depth = bbox_pts_3d[i, 0]

                # since the point clouds distance start from 2m, so 510/2 = 255
                color = cmap[int(510 / depth), :]

                x = int(np.round(bbox_pts_2d[i, 0]))
                y = int(np.round(bbox_pts_2d[i, 1]))
                cv2.circle(img, (x, y), 2, color=tuple(color), thickness=1)
            
            inlier_depths = filter_outliers(bbox_pts_3d[:, 0])
            best_distance = get_best_distance(inlier_depths, technique="average")
            cv2.putText(img, '{0:.2f} m'.format(best_distance), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        return img

    def pipeline (self, image, point_cloud, yolo):
        """ Build a Pipeline for a pair of 2 Calibrated Images"""
        img = image.copy()
        # Run obstacle detection in 2D
        result, pred_bboxes = run_obstacle_detection(img, yolo)
        # Fuse Point Clouds & Bounding Boxes
        img_final = self.lidar_camera_fusion(pred_bboxes, result, point_cloud)
        return img_final




def rectContains(pt, x, y, w, h, shrink_factor = 0):
    """Return a mask, that mask out points outside a shrink bounding box"""
    shrink_w = w * (1 - shrink_factor)
    shrink_h = h * (1 - shrink_factor)
    # remove points that are not inside the bbox
    top_left_x = int(x - shrink_w * 0.5)
    top_left_y = int(y - shrink_h * 0.5)
    bottom_right_x = int(x + shrink_w * 0.5)
    bottom_right_y = int(y + shrink_h * 0.5)
    mask = (pt[:, 0] >= top_left_x) & (pt[:, 0] < bottom_right_x) & (pt[:, 1] >= top_left_y) & (pt[:, 1] < bottom_right_y)
    return mask

def filter_outliers(depths):
    """ remove the outliers according to One Sigma"""
    inliers = []
    mu = statistics.mean(depths)
    std = statistics.stdev(depths)

    for d in depths:
        # one sigma
        if abs(d - mu) < std:
            inliers.append(d)
    return inliers

def get_best_distance(distances, technique="closest"):
    """get the Best Distance according to at least 3 criterias of your choice (closest, average, median, farthest, ...)"""
    if technique == "closest":
        return min(distances)
    elif technique == "average":
        return statistics.mean(distances)
    elif technique == " random":
        return random.choice(distances)
    else:
        return statistics.median(sorted(distances))


def comparing(image_files, index, prediction):
    """Comparing with the Ground Truth"""
    image_gt = cv2.cvtColor(cv2.imread(image_files[index]), cv2.COLOR_BGR2RGB).copy()

    with open(label_files[index], 'r') as f:
        fin = f.readlines()
        for line in fin:
            if line.split(" ")[0] != "DontCare":
                #print(line)
                x1_value = int(float(line.split(" ")[4]))
                y1_value = int(float(line.split(" ")[5]))
                x2_value = int(float(line.split(" ")[6]))
                y2_value = int(float(line.split(" ")[7]))
                dist = float(line.split(" ")[13])
                cv2.rectangle(image_gt, (x1_value, y1_value), (x2_value, y2_value), (0,205,0), 10)
                cv2.putText(image_gt, str(dist), (int((x1_value+x2_value)/2),int((y1_value+y2_value)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)    

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,20))
    ax1.imshow(image_gt)
    ax1.set_title('Ground Truth', fontsize=30)
    ax2.imshow(prediction) # or flag
    ax2.set_title('Prediction', fontsize=30)


def save_video_result(images, point_clouds, calib_file, result_path, yolo):
    """Saving the prediction result as video"""
    # Build a LiDAR2Cam object
    lidar2cam_video = LiDAR2Camera(calib_file)

    result_video = []
    for idx, img in enumerate(images):
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        point_cloud = np.asarray(o3d.io.read_point_cloud(point_clouds[idx]).points)
        result_video.append(lidar2cam_video.pipeline(image, point_cloud, yolo))

    out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (image.shape[1],image.shape[0]))
    
    for i in range(len(result_video)):
        out.write(cv2.cvtColor(result_video[i], cv2.COLOR_BGR2RGB))
        #out.write(result_video[i])
    out.release()



if __name__ == '__main__':
    # Load the Files
    image_files = sorted(glob.glob("data/img/*.png"))
    point_files = sorted(glob.glob("data/velodyne/*.pcd"))
    label_files = sorted(glob.glob("data/label/*.txt"))
    calib_files = sorted(glob.glob("data/calib/*.txt"))

    # first image and point cloud
    index = 0
    image = cv2.cvtColor(cv2.imread(image_files[index]), cv2.COLOR_BGR2RGB)
    cloud = o3d.io.read_point_cloud(point_files[index])
    points= np.asarray(cloud.points)

    # load YOLOv4
    yolo = YOLOv4(tiny=False)
    yolo.classes = "Yolov4/coco.names"
    yolo.make_model()
    yolo.load_weights("Yolov4/yolov4.weights", weights_type="yolo")

    # initialize LiDAR2Camera object 
    lidar2cam = LiDAR2Camera(calib_files[index])

    # display result for a single image
    plt.figure(figsize=(14,7))
    final_result = lidar2cam.pipeline(image.copy(), points, yolo)
    plt.imshow(final_result)
    plt.show()

    # compare prediction with ground_truth
    comparing(image_files, index, final_result)

    # output a result video
    video_images = sorted(glob.glob("videos/video1/images/*.png"))
    video_points = sorted(glob.glob("videos/video1/points/*.pcd"))
    calib_file = calib_files[index]
    result_path = 'early_fusion_out.avi'
    save_video_result(video_images, video_points, calib_file, result_path, yolo)