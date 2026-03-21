import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils.preprocessing import get_transforms, Vocabulary
import random

class ImageCaptionDataset(Dataset):
    def __init__(self, data_root, split="train", vocab=None, transform=None, debug=False, freq_threshold=5):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.debug = debug
        self.freq_threshold = freq_threshold
        
        # Determine paths
        # Assuming format:
        # data_root/
        #   images/
        #     train/ or train2014/
        #     val/ or val2014/
        #   captions/
        #     captions_*.json or annotations_*.json (with optional year suffix 2014)
        
        # Find available image directories
        image_root = os.path.join(data_root, "images")
        available_img_dirs = []
        if os.path.exists(image_root):
            for item in os.listdir(image_root):
                item_path = os.path.join(image_root, item)
                if os.path.isdir(item_path):
                    available_img_dirs.append(item_path)
        
        # Use the first available image directory (prefer requested split, then any available)
        self.img_dir = None
        if available_img_dirs:
            # First try the requested split
            requested_dir = os.path.join(image_root, split)
            if requested_dir in available_img_dirs:
                self.img_dir = requested_dir
            else:
                # Use any available directory
                self.img_dir = available_img_dirs[0]
                print(f"⚠️  Requested image dir '{split}' not found, using '{os.path.basename(self.img_dir)}'")
        
        if self.img_dir is None:
            # Fallback to requested split (will create dummy images if needed)
            self.img_dir = os.path.join(image_root, split)
        
        # Try multiple naming conventions for captions
        # 1. Simple format: captions_train.json
        # 2. COCO format: captions_train2014.json
        # 3. Annotations prefix: annotations_train.json
        # 4. Annotations with year: annotations_train2014.json
        possible_paths = [
            os.path.join(data_root, "captions", f"captions_{split}.json"),
            os.path.join(data_root, "captions", f"captions_{split}2014.json"),
            os.path.join(data_root, "captions", f"annotations_{split}.json"),
            os.path.join(data_root, "captions", f"annotations_{split}2014.json"),
        ]
        
        self.caption_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.caption_path = path
                break
        
        if self.caption_path is None:
            # Fallback to first option (will trigger dummy data warning)
            self.caption_path = possible_paths[0]
        
        self.data = self._load_data()
        
        # Take small subset for debug mode
        if self.debug:
            self.data = self.data[:100]
            
        self.vocab = vocab
        if self.vocab is None and split == "train":
            self.vocab = Vocabulary(self.freq_threshold)
            self.vocab.build_vocabulary([item["caption"] for item in self.data])
            
    def _load_data(self):
        """Loads data from JSON. Supports two formats:
        
        1. Simple format (list):
           [
               {"image_id": "img1.jpg", "caption": "A dog catching frisbee"},
               ...
           ]
        
        2. COCO format (dict with images and annotations):
           {
               "images": [{"id": 1, "file_name": "COCO_train2014_000000000009.jpg"}, ...],
               "annotations": [
                   {"image_id": 1, "caption": "A person doing a trick on a truck"},
                   ...
               ]
           }
        
        If data doesn't exist, generates dummy data.
        """
        # First try the requested split
        captions_loaded = False
        if os.path.exists(self.caption_path):
            print(f"Loading captions from: {self.caption_path}")
            with open(self.caption_path, "r") as f:
                raw_data = json.load(f)
            captions_loaded = True
        else:
            # If requested split not found, try alternatives
            print(f"⚠️  {self.caption_path} not found")
            alt_split = "val" if self.split == "train" else "train"
            alt_paths = [
                os.path.join(self.data_root, "captions", f"captions_{alt_split}.json"),
                os.path.join(self.data_root, "captions", f"captions_{alt_split}2014.json"),
                os.path.join(self.data_root, "captions", f"annotations_{alt_split}.json"),
                os.path.join(self.data_root, "captions", f"annotations_{alt_split}2014.json"),
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"   Using alternative: {alt_path}")
                    self.caption_path = alt_path
                    with open(self.caption_path, "r") as f:
                        raw_data = json.load(f)
                    captions_loaded = True
                    break
        
        if not captions_loaded:
            print(f"Warning: No captions found. Using DUMMY data.")
            return self._generate_dummy_data()
        
        # Handle COCO format: {"images": [...], "annotations": [...]}
        if isinstance(raw_data, dict) and "annotations" in raw_data:
            print(f"Detected COCO format with {len(raw_data.get('annotations', []))} annotations")
            
            # Build image_id -> filename mapping
            image_map = {}
            for img in raw_data.get("images", []):
                image_map[img["id"]] = img.get("file_name", f"{img['id']}.jpg")
            
            # Convert to simple format: list of {image_id, caption}
            converted_data = []
            for ann in raw_data.get("annotations", []):
                img_id = ann["image_id"]
                converted_data.append({
                    "image_id": image_map.get(img_id, f"{img_id}.jpg"),
                    "caption": ann.get("caption", "")
                })
            return converted_data
        
        # Handle simple format: list of dicts
        elif isinstance(raw_data, list):
            print(f"Detected simple format with {len(raw_data)} items")
            return raw_data
        
        else:
            print(f"Warning: Unknown format. Using DUMMY data.")
            return self._generate_dummy_data()
    
    def _generate_dummy_data(self):
        """Generate dummy data for testing."""
        os.makedirs(self.img_dir, exist_ok=True)
        dummy_data = []
        for i in range(100 if self.debug else 500):
            img_name = f"dummy_{i}.jpg"
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                # Create a random RGB image
                Image.new('RGB', (224, 224), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255))).save(img_path)
            
            dummy_data.append({
                "image_id": img_name,
                "caption": f"This is dummy image {i} mostly for testing."
            })
        return dummy_data
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item["image_id"]
        caption = item["caption"]
        
        img_path = os.path.join(self.img_dir, img_name)
        
        # If image doesn't exist in current directory, skip this item
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            # Return a dummy image instead of crashing
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        else:
            image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = get_transforms(is_train=False)(image)
            
        # Numericalize caption
        numericalized_caption = [self.vocab.stoi[self.vocab.start_token]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi[self.vocab.end_token])
        
        return image, torch.tensor(numericalized_caption), caption


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        captions = [item[2] for item in batch]
        
        # Pad targets
        from torch.nn.utils.rnn import pad_sequence
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        
        return imgs, targets, captions


def get_loaders(config):
    # Transforms
    train_transform = get_transforms(image_size=config["image_size"], is_train=True)
    val_transform = get_transforms(image_size=config["image_size"], is_train=False)
    
    # Datasets
    train_dataset = ImageCaptionDataset(
        data_root=config["data_dir"],
        split="train",
        transform=train_transform,
        debug=config["debug"],
        freq_threshold=config["min_word_freq"]
    )
    
    pad_idx = train_dataset.vocab.stoi[train_dataset.vocab.pad_token]
    batch_size = config["debug_batch_size"] if config["debug"] else config["batch_size"]
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        shuffle=True,
        collate_fn=MyCollate(pad_idx=pad_idx),
        drop_last=True
    )
    
    val_loader = None
    if not config["debug"]:
        val_dataset = ImageCaptionDataset(
            data_root=config["data_dir"],
            split="val",
            vocab=train_dataset.vocab,
            transform=val_transform,
            debug=False,
            freq_threshold=config["min_word_freq"]
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=config["num_workers"],
            shuffle=False,
            collate_fn=MyCollate(pad_idx=pad_idx)
        )
        
    return train_loader, val_loader, train_dataset.vocab
