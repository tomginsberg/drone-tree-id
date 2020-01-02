import detectron2
from detectron2.utils.visualizer import Visualizer
import cv2
from register_shapefile_datatset import get_dataset_dicts
from detectron2.data import MetadataCatalog

classes, dataset_dicts = get_dataset_dicts("/Users/Ro/Google Drive/UBC_EngCapstone/sample_data/CPT2a-n")
img = cv2.imread('/Users/Ro/Google Drive/UBC_EngCapstone/sample_data/CPT2a-n/CPT2a-n_ortho-resample.tif')
visualizer = Visualizer(img[:, :, :], metadata=MetadataCatalog.get("CPT2a-n"), scale=1)
vis = visualizer.draw_dataset_dict(dataset_dicts[0])
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',vis.get_image()[:, :, :])
cv2.waitKey(0)
