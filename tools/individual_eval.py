import logging
import os
from collections import OrderedDict

import wandb

import detectron2.utils.comm as comm
from deepent.config import add_deepent_config
from deepent.data.register_datasets import register_datasets
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results, inference_on_dataset, \
    print_csv_format, DatasetEvaluator
from detectron2.utils.events import CommonMetricPrinter, JSONWriter
from detectron2.utils.logger import setup_logger
from tools.wandb_writer import WandbWriter
import glob

def ind_eval(args):
    #create new mini-datasets
    test_tiles = []
    test_tile_segs = []
    for location in glob.glob('/home/ubuntu/RGBD-Tree-Segs-Clean/test/*'):
        filenames = glob.glob(location + '/*.png')
        with open(os.path.join(location,'segs.json'),'r') as segfile:
            segs = json.load(segfile)
            for ind in np.random.randint(0,len(filenames),2):
                test_tiles.append(filenames[ind])
                test_tile_segs.append(list(filter(lambda x : x["file_name"]==filenames[ind],segs)))
                
    test_dir = '/home/ubuntu/RGBD-Tree-Segs-Clean/test/'                
    folder_names = []                 
    for i,path in enumerate(test_tiles):
        folder_names.append('temporary_'+str(i))
        try:
            os.mkdir(test_dir + folder_names[-1])
            shutil.copy(path, test_dir + folder_names[-1] + '/' + path.split('/')[-1])
        except:
            print("Error creating temporary dataset directories!")
            return
        with open(os.path.join(test_dir,folder_names[-1],'segs.json'),'w') as segfile:
            segfile.write(json.dumps(test_tile_segs[i]))
            
    test_set_names = list(map(lambda x : x + "_test", folder_names))        
    register_datasets(f'/home/ubuntu/RGBD-Tree-Segs-Clean/')
    
    cfg = get_cfg()
    add_deepent_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TEST = test_set_names
    cfg.freeze()
    default_setup(cfg, args)
    
    #Run Eval
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    res = Trainer.test(cfg, model,wandb_on=False)
    if comm.is_main_process():
                verify_results(cfg, res)
        
    #Visualize with different models and show evals
    predictor = RGBDPredictor(cfg)
    fig,axes = plt.subplots(10,2,figsize=(20,70))
    ax = axes.ravel()
    i=0
    for test_set_name in test_set_names:
        data = list(DatasetCatalog.get(test_set_name))
        metadata = MetadataCatalog.get(test_set_name)
        dic=data[0]
        rgba = cv2.imread(dic["file_name"], cv2.IMREAD_UNCHANGED)
        img = cv2.imread(dic["file_name"])
        predictions = predictor(rgba)
        visualizer = Visualizer(img, metadata=metadata)
        vis = visualizer.draw_instance_predictions(predictions["instances"].to("cpu")).get_image()
        name = res[test_set_name]["segm"]["AP"]
        ax[i].set_title("AP: " + str(name))
        ax[i].axis('off')
        ax[i].imshow(vis)
        i+=1
        visualizer2 = Visualizer(img, metadata=metadata)
        vis2 = visualizer2.draw_dataset_dict(dic).get_image()
        ax[i].set_title('Annotation')
        ax[i].axis('off')
        ax[i].imshow(vis2)
        i+=1
        
    #remove temporary dirs
    os.chdir('/home/ubuntu/RGBD-Tree-Segs-Clean/test')
    for folder in glob.glob('temporary*'):
        for file in glob.glob(folder + '/*'):
            os.remove(file)
        os.rmdir(folder)   
        
    return res

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    ind_eval(args)