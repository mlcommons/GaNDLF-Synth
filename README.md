Some basic info:
  - Deadline for package is 10th of October 2024 (MICCAI)
  - POC by the end of August 2024, allow new developer commits in September or so


Dev requirements/what to keep in mind:
  - do we need any additional CLI entrypoints? should any existing ones be extended/modified?
  - for now our focus is to have one prototype of architecture working (AE, GAN and Diffusion model)
  - also, a note to keep for us regarding the architecture specific pipelines - there are GANS for style transfer, i.e. CycleGAN and they need pairs of images to work


* Szymon:
I believe we need the following entrypoint scripts:
 - construct_csv (I think it can be used nearly 1:1 except for lable generation)
 - preprocess
 - run (here we need to change the input pipeline and empoly our custom trainign manager)
 - recover_config (compat with synth specific config)
 - config_generator (same as above)
 - update_version 
 - verify_install
 - split_csv (here we need to think on options to give, like if someone wants only some test set, or test + validation etc)