"""Optional historical full-tile perceptual render; photometric render unchanged."""


def image_for_perceptual(image, renderer, camera, model, pipe, background,
                         requested_mult, photo_mult):
    if requested_mult is None or requested_mult == photo_mult:
        return image, False
    return renderer(camera, model, pipe, background, requested_mult, 1.0)['render'], True
