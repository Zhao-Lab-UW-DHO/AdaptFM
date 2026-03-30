class AnnotationSession:
    def __init__(self, image_paths):
        self.image_paths = list(image_paths)
        self.idx = 0

    def set_images(self, image_paths):
        self.image_paths = list(image_paths)
        self.idx = 0

    def has_images(self):
        return len(self.image_paths) > 0

    def current_path(self):
        return self.image_paths[self.idx]

    def next(self):
        if self.idx < len(self.image_paths) - 1:
            self.idx += 1
            return self.current_path()
        return None

    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            return self.current_path()
        return None
