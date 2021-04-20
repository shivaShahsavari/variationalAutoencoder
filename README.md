# variational Autoencoder
A variational autoencoder is similar to a regular autoencoder except that it is a generative model. This “generative” aspect stems from placing an additional constraint on the loss function such that the latent space is spread out and doesn’t contain dead zones where reconstructing an input from those locations results in garbage. By doing this, we can randomly sample a vector from the latent space and hopefully create a meaninful decoded output from it.  
The “variational” part comes from the fact that we’re trying to approximate the posterior distribution pθ(z|x) with a variational distribution qϕ(z|x). Thus, the encoder outputs parameters to this variational distribution which is just a multivariate Gaussian distribution, and the latent representation is obtained by then sampling this distribution. The decoder then takes the latent representation and tries to reconstruct the original input from it.  

<img src="image/VAE_architecture.png" width="800" height="450">  

Applications of variational Autoencoder: 
* Dimensionality Reduction
* Image Compression
* Image Denoising
* Feature Extraction
* Image generation
* Sequence to sequence prediction
* Recommendation system



sources:
https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
http://alexadam.ca/ml/2017/05/05/keras-vae.html
https://www.jeremyjordan.me/variational-autoencoders/

