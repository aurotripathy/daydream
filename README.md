# DayDream

DayDream is a gradio app wrapped over [DreamBooth](https://arxiv.org/abs/2208.12242), a method to personalize text2image models like stable diffusion given just a few (3~5) images of a subject. 

### Elon blue finch Example

My assumption is that you know how to install and run the basic HF dreambooth code.  

To generate the fine-tuned model, run the script `elon-run.sh`

The necessary 'seeding' images are at  [images/elon/](images/elon/)

Once fine-tuning is complete, run the gradio-abded app [gradio/dream-ui.py](gradio/dream-ui.py)

Your prompt should be something like `A photo of an sks person holding a blue finch`.

You display should look liek this: [output](assets/A%20photo%20of%20a%20sks%20person%20holding%20a%20blue%20finch.png)
