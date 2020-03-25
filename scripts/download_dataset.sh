echo "This script downloads datasets used in the DVR project."
echo "Choose from the following options:"
echo "0 - ShapeNet dataset with Choy et. al. renderings for 2.5D supervised models (single-view reconstruction)"
echo "1 - ShapeNet dataset with Kato et. al. renderings for 2D supervised models (single-view reconstruction)"
echo "2 - DTU subset (multi-view reconstruction)"
read -p "Enter dataset ID you want to download: " ds_id

if [ $ds_id == 0 ]
then
    echo "You chose 0: ShapeNet dataset with Choy et. al. renderings for 2.5D supervised models (single-view reconstruction)"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/ShapeNet.zip
    echo "done! Start unzipping ..."
    unzip ShapeNet.zip
    echo "done!"
elif [ $ds_id == 1 ]
then
    echo "You chose 1: ShapeNet dataset with Kato et. al. renderings for 2D supervised models (single-view reconstruction)"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
    echo "done! Start unzipping ..."
    unzip NMR_Dataset.zip
    echo "done!"
elif [ $ds_id == 2 ]
then
    echo "You chose 2: DTU subset (multi-view reconstruction)"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/DTU.zip
    echo "done! Start unzipping ..."
    unzip DTU.zip
    echo "done!"
else
    echo "You entered an invalid ID!"
fi
