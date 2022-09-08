from dalle import generate_images


def get_images(prompts, n_predictions = 8, gen_top_k = None, gen_top_p = None, temperature = None, cond_scale = 10.0):
    images = generate_images(prompts, n_predictions = 8, gen_top_k = None, gen_top_p = None, temperature = None, cond_scale = 10.0)
    return images