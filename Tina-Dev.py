bl_info = {
        'name': 'Tina (dev mode)',
        'description': 'A soft-renderer based on Taichi programming language',
        'author': 'archibate <1931127624@qq.com>',
        'version': (0, 0, 0),
        'blender': (2, 81, 0),
        'location': 'Render -> Tina',
        'support': 'TESTING',
        'wiki_url': 'https://github.com/archibate/ptina/wiki',
        'tracker_url': 'https://github.com/archibate/ptina/issues',
        'warning': 'Development mode',
        'category': 'Render',
}


import sys
sys.path.insert(0, '/home/bate/Develop/cristina')


registered = False


def register():
    print('Tina-Dev register...')
    import ptina
    ptina.register()

    global registered
    registered = True
    print('...register done')


def unregister():
    print('Tina-Dev unregister...')
    import ptina
    ptina.unregister()

    global registered
    registered = False
    print('...unregister done')


def reload_addon():
    if registered:
        import ptina
        ptina.unregister()
        del ptina
    mods_to_del = []
    for k in sys.modules:
        if k.startswith('ptina.') or k == 'ptina':
            mods_to_del.append(k)
    for k in mods_to_del:
        sys.modules.pop(k)
    import ptina
    ptina.register()


@eval('lambda x: x()')
def _():
    class Reload:
        def __repr__(self):
            import os
            import bpy
            os.system('clear')
            reload_addon()
            bpy.context.scene.frame_current = bpy.context.scene.frame_current
            return 'reloaded'

    __import__('bpy').a = Reload()
