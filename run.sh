# TODO: GCS command to download the folder first
# gcloud storage rsync -r gs://tour_storage/data/tandt/truck data/tandt/truck


gcloud storage rsync -r gs://tour_storage/data/examples/kitchen data/examples/kitchen

python tools/reconstruct.py --image-dir data/examples/kitchen/images/

gcloud storage rsync -r data/examples/kitchen/images/ gs://tour_storage/data/examples/kitchen/images/