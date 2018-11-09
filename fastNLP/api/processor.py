

class Processor:
    def __init__(self, field_name, new_added_field_name):
        self.field_name = field_name
        if new_added_field_name is None:
            self.new_added_field_name = field_name
        else:
            self.new_added_field_name = new_added_field_name

    def process(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)