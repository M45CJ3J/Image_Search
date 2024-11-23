from PIL import Image
from pathlib import Path
import numpy as np
from feature_extractor import FeatureExtractor

if __name__ == '__main__':
    fe = FeatureExtractor() 

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg

        feature = fe.extract(img=Image.open(img_path))
        
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        print(feature_path)

        np.save(feature_path,feature)