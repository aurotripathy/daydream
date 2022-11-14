embed="elon"
echo "Embedding $embed"

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR='/media/auro/NLP/diffusers/diffusers/examples/dreambooth/images/'$embed
export OUTPUT_DIR=$embed'_saved_model'
export CLASS_DIR='path_to_'$embed'_class_images'

echo model name: $MODEL_NAME
echo instance dir: $INSTANCE_DIR
echo output dir: $OUTPUT_DIR
echo class dir: $CLASS_DIR

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
