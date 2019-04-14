name=$1

mkdir extensions/$name
mkdir extensions/$name/{dataloaders,runners,models,scripts}
touch extensions/$name/config.yaml
