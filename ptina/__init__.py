bl_info = {
        'name': 'Taichi Tina',
        'description': 'A soft renderer based on Taichi programming language',
        'author': 'archibate <1931127624@qq.com>',
        'version': (0, 1, 1),
        'blender': (2, 90, 0),
        'location': 'Render -> Tina Render',
        'support': 'COMMUNITY',
        'wiki_url': 'https://github.com/taichi-dev/taichi_three/wiki',
        'tracker_url': 'https://github.com/taichi-dev/taichi_three/issues',
        'category': 'Render',
}


def register():
    from . import blender
    return blender.register()


def unregister():
    from . import blender
    return blender.unregister()
