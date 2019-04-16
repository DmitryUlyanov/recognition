


# -------------
# SMILE 
# -------------
p="data/C2ST_diana/smile"

# DFI
# what=dfi
# mkdir -p $p/splits/${what}
# python scripts/C2ST_diana/train_test_split.py \
#                 --dir_class0 $p/results-dfi \
#                 --dir_class1 $p/celeba_smiles \
#                 --save_dir $p/splits/${what} \
#                 --same_size

# # Smart
# what=l50
# mkdir -p $p/splits/${what}
# python scripts/C2ST_diana/train_test_split.py \
#                 --dir_class0 $p/results-l50 \
#                 --dir_class1 $p/celeba_smiles \
#                 --save_dir $p/splits/${what} \
#                 --same_size

# # Plane VGG
# what=plain
# mkdir -p $p/splits/${what}
# python scripts/C2ST_diana/train_test_split.py \
#                 --dir_class0 $p/results-l50-planevgg \
#                 --dir_class1 $p/celeba_smiles \
#                 --save_dir $p/splits/${what} \
#                 --same_size


# # DCGAN
# what=dcgan
# mkdir -p $p/splits/${what}
# python scripts/C2ST_diana/train_test_split.py \
#                 --dir_class0 $p/results-dcgan \
#                 --dir_class1 $p/celeba_smiles \
#                 --save_dir $p/splits/${what} \
#                 --same_size

# DCGAN
# what=cyclegan
# mkdir -p $p/splits/${what}
# python scripts/C2ST_diana/train_test_split.py \
#                 --dir_class0 $p/results-cyclegan \
#                 --dir_class1 $p/celeba_smiles \
#                 --save_dir $p/splits/${what} \
#                 --same_size

# -------------
# M-F 
# -------------
p="data/C2ST_diana/m-f"

# DFI
# what=dfi
# mkdir -p $p/splits/${what}
# python scripts/C2ST_diana/train_test_split.py \
#                 --dir_class0 $p/dfi \
#                 --dir_class1 $p/celeba_female \
#                 --save_dir $p/splits/${what} \
#                 --same_size

# # Smart
# what=l50
# mkdir -p $p/splits/${what}
# python scripts/C2ST_diana/train_test_split.py \
#                 --dir_class0 $p/resuls-vgg-smart-m-f \
#                 --dir_class1 $p/celeba_female \
#                 --save_dir $p/splits/${what} \
#                 --same_size


# # Plane VGG
# what=plain
# mkdir -p $p/splits/${what}
# python scripts/C2ST_diana/train_test_split.py \
#                 --dir_class0 $p/resuls-vgg-plain-m-f \
#                 --dir_class1 $p/celeba_female \
#                 --save_dir $p/splits/${what} \
#                 --same_size


# # DCGAN
# what=dcgan
# mkdir -p $p/splits/${what}
# python scripts/C2ST_diana/train_test_split.py \
#                 --dir_class0 $p/resuls-dcgan-m-f \
#                 --dir_class1 $p/celeba_female \
#                 --save_dir $p/splits/${what} \
#                 --same_size


# what=cyclegan
# mkdir -p $p/splits/${what}
# python scripts/C2ST_diana/train_test_split.py \
#                 --dir_class0 $p/resuls-cyclegan \
#                 --dir_class1 $p/celeba_female \
#                 --save_dir $p/splits/${what} \
#                 --same_size


# -------------
# Brown 
# -------------

p="data/C2ST_diana/brown"

# DFI
what=dfi
mkdir -p $p/splits/${what}
python scripts/C2ST_diana/train_test_split.py \
                --dir_class0 $p/dfi \
                --dir_class1 $p/celeba_brown \
                --save_dir $p/splits/${what} \
                --same_size

# Smart
what=l50
mkdir -p $p/splits/${what}
python scripts/C2ST_diana/train_test_split.py \
                --dir_class0 $p/resuls-vgg-smart-brown \
                --dir_class1 $p/celeba_brown \
                --save_dir $p/splits/${what} \
                --same_size


# Plane VGG
what=plain
mkdir -p $p/splits/${what}
python scripts/C2ST_diana/train_test_split.py \
                --dir_class0 $p/resuls-vgg-plain-brown \
                --dir_class1 $p/celeba_brown \
                --save_dir $p/splits/${what} \
                --same_size


# DCGAN
what=dcgan
mkdir -p $p/splits/${what}
python scripts/C2ST_diana/train_test_split.py \
                --dir_class0 $p/resuls-dcgan-brown \
                --dir_class1 $p/celeba_brown \
                --save_dir $p/splits/${what} \
                --same_size


what=cyclegan
mkdir -p $p/splits/${what}
python scripts/C2ST_diana/train_test_split.py \
                --dir_class0 $p/resuls-cyclegan \
                --dir_class1 $p/celeba_brown \
                --save_dir $p/splits/${what} \
                --same_size