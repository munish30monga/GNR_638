mkdir -p datasets
cd datasets
mkdir -p cub
cd cub
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1 -O cub.tgz
tar -xvzf cub.tgz
rm cub.tgz