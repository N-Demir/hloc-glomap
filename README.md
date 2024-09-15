# lightglue-glomap

### first create the image dataset and folder
`pixi run video-processing --data $DATA-PATH --output-dir $OUTPUT-PATH`
### Reconstruct from image data
`pixi run reconstruct --image-dir $OUTPUT-PATH/images`
### Register new iamges
Requires having a vocab tree guide [here](https://colmap.github.io/faq.html#register-localize-new-images-into-an-existing-reconstruction)
### Train Splat
`DATA_DIR="$OUTPUT-PATH" pixi run train-splat `
### Cleanup splat
### Upload to Playcanvas
### Show on the gradio app