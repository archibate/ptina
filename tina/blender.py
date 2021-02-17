import bpy
import bgl
import numpy as np
import mtworker

from tina.multimesh import compose_multiple_meshes


worker = None


def calc_camera_matrices(depsgraph):
    camera = depsgraph.scene.camera
    render = depsgraph.scene.render
    scale = render.resolution_percentage / 100.0
    proj = np.array(camera.calc_matrix_camera(depsgraph,
        x=render.resolution_x * scale, y=render.resolution_y * scale,
        scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y))
    view = np.linalg.inv(np.array(camera.matrix_world))
    return view, proj


def bmesh_verts_to_numpy(bm):
    arr = [x.co for x in bm.verts]
    if len(arr) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(arr, dtype=np.float32)


def bmesh_faces_to_numpy(bm):
    arr = [[e.index for e in f.verts] for f in bm.faces]
    if len(arr) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(arr, dtype=np.int32)


def bmesh_face_norms_to_numpy(bm):
    vnorms = [x.normal for x in bm.verts]
    if len(vnorms) == 0:
        vnorms = np.zeros((0, 3), dtype=np.float32)
    else:
        vnorms = np.array(vnorms)
    norms = [
        [vnorms[e.index] for e in f.verts]
        if f.smooth else [f.normal for e in f.verts]
        for f in bm.faces]
    if len(norms) == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)
    return np.array(norms, dtype=np.float32)


def bmesh_face_coors_to_numpy(bm):
    uv_lay = bm.loops.layers.uv.active
    if uv_lay is None:
        return np.zeros((len(bm.faces), 3, 2), dtype=np.float32)
    coors = [[l[uv_lay].uv for l in f.loops] for f in bm.faces]
    if len(coors) == 0:
        return np.zeros((0, 3, 2), dtype=np.float32)
    return np.array(coors, dtype=np.float32)


def blender_get_object_mesh(object, depsgraph=None):
    import bmesh
    bm = bmesh.new()
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    object_eval = object.evaluated_get(depsgraph)
    bm.from_object(object_eval, depsgraph)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    verts = bmesh_verts_to_numpy(bm)[bmesh_faces_to_numpy(bm)]
    norms = bmesh_face_norms_to_numpy(bm)
    coors = bmesh_face_coors_to_numpy(bm)
    return verts, norms, coors


class TinaLightPanel(bpy.types.Panel):
    '''Tina light options'''

    bl_label = 'Tina Light'
    bl_idname = 'DATA_PT_tina'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'data'

    def draw(self, context):
        layout = self.layout
        object = context.object

        if object.type == 'LIGHT':
            layout.prop(object.data, 'color')
            layout.prop(object.data, 'energy')
            if object.data.type == 'POINT':
                layout.prop(object.data, 'shadow_soft_size', text='Radius')
            elif object.data.type == 'AREA':
                layout.prop(object.data, 'size')


class TinaWorldPanel(bpy.types.Panel):
    '''Tina world options'''

    bl_label = 'Tina World'
    bl_idname = 'WORLD_PT_tina'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'world'

    def draw(self, context):
        layout = self.layout
        world = context.scene.world

        layout.prop(world, 'tina_color')
        layout.prop(world, 'tina_strength')


class TinaRenderPanel(bpy.types.Panel):
    '''Tina render options'''

    bl_label = 'Tina Render'
    bl_idname = 'RENDER_PT_tina'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'

    def draw(self, context):
        layout = self.layout
        options = context.scene.tina_render

        layout.prop(options, 'render_samples')
        layout.prop(options, 'viewport_samples')
        layout.prop(options, 'start_pixel_size')


class TinaRenderEngine(bpy.types.RenderEngine):
    # These three members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = "TINA"
    bl_label = "Tina"
    bl_use_preview = True

    # Init is called whenever a new render engine instance is created. Multiple
    # instances may exist at the same time, for example for a viewport and final
    # render.
    def __init__(self):
        self.scene_data = None
        self.draw_data = None

        self.object_to_mesh = {}
        self.object_to_light = {}
        self.nblocks = 0
        self.nsamples = 0
        self.viewport_samples = 16

    # When the render engine instance is destroy, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    def __del__(self):
        pass

    def __add_mesh_object(self, object, depsgraph):
        print('[TinaBlend] adding mesh object', object.name)

        verts, norms, coors = blender_get_object_mesh(object, depsgraph)
        world = np.array(object.matrix_world)

        mtlid = -1
        self.object_to_mesh[object] = world, verts, norms, coors, mtlid

    def __add_light_object(self, object, depsgraph):
        print('[TinaBlend] adding light object', object.name)

        world = np.array(object.matrix_world)
        color = np.array(object.data.color)
        color *= object.data.energy
        type = object.data.type

        if type == 'POINT':
            size = max(object.data.shadow_soft_size, 1e-6)
            color /= 2 * np.pi * size**2
        elif type == 'AREA':
            assert object.data.shape == 'SQUARE'
            size = max(object.data.size, 1e-6)
        else:
            raise ValueError(type)

        self.object_to_light[object] = world, color, size, type

    def __setup_scene(self, depsgraph):
        self.update_stats('Initializing', 'Loading scene')

        scene = depsgraph.scene
        options = scene.tina_render

        for object in depsgraph.ids:
            if isinstance(object, bpy.types.Object):
                if object.type == 'MESH':
                    self.__add_mesh_object(object, depsgraph)
                elif object.type == 'LIGHT':
                    self.__add_light_object(object, depsgraph)

        self.__on_update(depsgraph)

    def __update_scene(self, depsgraph):
        need_update = False
        for update in depsgraph.updates:
            object = update.id

            if isinstance(object, bpy.types.Scene):
                obj_to_del = []
                for obj in self.object_to_mesh:
                    if obj.name not in object.objects:
                        print('[TinaBlend] removing mesh object', obj)
                        obj_to_del.append(obj)
                for obj in obj_to_del:
                    del self.object_to_mesh[obj]
                    need_update = True

                obj_to_del = []
                for obj in self.object_to_light:
                    if obj.name not in object.objects:
                        print('[TinaBlend] removing light object', obj)
                        obj_to_del.append(obj)
                for obj in obj_to_del:
                    del self.object_to_light[obj]
                    need_update = True

            if isinstance(object, bpy.types.Object):
                if object.type == 'MESH':
                    self.__add_mesh_object(object, depsgraph)
                    need_update = True
                elif object.type == 'LIGHT':
                    self.__add_light_object(object, depsgraph)
                    need_update = True

        if need_update:
            self.update_stats('Initializing', 'Updating scene')

            self.__on_update(depsgraph)

    def __on_update(self, depsgraph):
        meshes = []
        for world, verts, norms, coors, mtlid in self.object_to_mesh.values():
            meshes.append((verts, norms, coors, world, mtlid))
        vertices, mtlids = compose_multiple_meshes(meshes)

        worker.load_model(vertices, mtlids)
        self.update_stats('Initializing', 'Constructing tree')
        worker.build_tree()

        self.update_stats('Initializing', 'Updating lights')
        worker.clear_lights()
        for world, color, size, type in self.object_to_light.values():
            worker.add_light(world, color, size, type)

        self.__reset_samples(depsgraph.scene)

    def __reset_samples(self, scene):
        self.nsamples = 0
        self.nblocks = scene.tina_render.start_pixel_size

    # This is the method called by Blender for both final renders (F12) and
    # small preview for materials, world and lights.
    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)
        view, proj = calc_camera_matrices(depsgraph)

        self.__setup_scene(depsgraph)
        self.__update_camera(proj @ view)

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.size_x, self.size_y)

        nsamples = scene.tina_render.render_samples
        for samp in range(nsamples):
            self.update_stats('Rendering', f'{samp}/{nsamples} Samples')
            self.update_progress((samp + .5) / nsamples)
            if self.test_break():
                break
            worker.render()
            img = worker.get_image()

            img = np.ascontiguousarray(img.swapaxes(0, 1))
            rect = img.reshape(self.size_x * self.size_y, 4).tolist()
            # import code; code.interact(local=locals())
            layer = result.layers[0].passes["Combined"]
            layer.rect = rect
            self.update_result(result)
        else:
            self.update_progress(1.0)

        self.end_result(result)

    def __update_camera(self, perspective):
        worker.set_camera(np.array(perspective))

    # For viewport renders, this method gets called once at the start and
    # whenever the scene or 3D viewport changes. This method is where data
    # should be read from Blender in the same thread. Typically a render
    # thread will be started to do the work while keeping Blender responsive.
    def view_update(self, context, depsgraph):
        print('[TinaBlend] view_update')

        region = context.region
        region3d = context.region_data
        view3d = context.space_data
        scene = depsgraph.scene

        # Get viewport dimensions
        dimensions = region.width, region.height
        perspective = region3d.perspective_matrix.to_4x4()
        self.size_x, self.size_y = dimensions

        if not self.scene_data:
            # First time initialization
            self.scene_data = True
            first_time = True

            # Loop over all datablocks used in the scene.
            print('[TinaBlend] setup scene')
            self.__setup_scene(depsgraph)
            self.__update_camera(perspective)
        else:
            first_time = False

            print('[TinaBlend] update scene')
            # Test which datablocks changed
            for update in depsgraph.updates:
                print("Datablock updated:", update.id.name)

            self.__update_scene(depsgraph)

            # Test if any material was added, removed or changed.
            if depsgraph.id_type_updated('MATERIAL'):
                print('[TinaBlend] Materials updated')

        # Loop over all object instances in the scene.
        if first_time or depsgraph.id_type_updated('OBJECT'):
            for instance in depsgraph.object_instances:
                pass

    # For viewport renders, this method is called whenever Blender redraws
    # the 3D viewport. The renderer is expected to quickly draw the render
    # with OpenGL, and not perform other expensive work.
    # Blender will draw overlays for selection and editing on top of the
    # rendered image automatically.
    def view_draw(self, context, depsgraph):
        print('[TinaBlend] view_draw')

        region = context.region
        region3d = context.region_data
        scene = depsgraph.scene
        max_samples = scene.tina_render.viewport_samples

        # Get viewport dimensions
        dimensions = region.width, region.height
        perspective = region3d.perspective_matrix.to_4x4()

        # Bind shader that converts from scene linear to display space,
        bgl.glEnable(bgl.GL_BLEND)
        bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
        self.bind_display_space_shader(scene)

        if not self.draw_data or self.draw_data.dimensions != dimensions \
                or self.nblocks != 0:
            width, height = dimensions
            if self.nblocks != 0:
                width //= self.nblocks
                height //= self.nblocks
            worker.set_size(width, height)

        if not self.draw_data or self.draw_data.dimensions != dimensions \
                or self.draw_data.perspective != perspective:
            self.__reset_samples(scene)
            self.__update_camera(perspective)

        if self.nsamples < max_samples:
            if self.nblocks > 1:
                self.nsamples = 0
                worker.clear()
            else:
                if self.nblocks == 1:
                    worker.clear()
                self.nsamples += 1

            self.update_stats('Rendering', f'{self.nsamples}/{max_samples} Samples')

            worker.render(self.nblocks == 0)
            self.draw_data = TinaDrawData(dimensions, perspective)

            if self.nsamples < max_samples or self.nblocks != 0:
                self.tag_redraw()

            self.nblocks //= 2

        self.draw_data.draw()

        self.unbind_display_space_shader()
        bgl.glDisable(bgl.GL_BLEND)


class TinaDrawData:
    def __init__(self, dimensions, perspective):
        print('[TinaBlend] redraw!')
        # Generate dummy float image buffer
        self.dimensions = dimensions
        self.perspective = perspective
        width, height = dimensions

        resx, resy = worker.get_size()

        pixels = np.empty(resx * resy * 3, np.float32)
        worker.fast_export_image(pixels)
        self.pixels = bgl.Buffer(bgl.GL_FLOAT, resx * resy * 3, pixels)

        # Generate texture
        self.texture = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenTextures(1, self.texture)
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
        bgl.glTexImage2D(bgl.GL_TEXTURE_2D, 0, bgl.GL_RGB16F, resx, resy, 0, bgl.GL_RGB, bgl.GL_FLOAT, self.pixels)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MIN_FILTER, bgl.GL_NEAREST)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_MAG_FILTER, bgl.GL_NEAREST)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_S, bgl.GL_CLAMP_TO_EDGE)
        bgl.glTexParameteri(bgl.GL_TEXTURE_2D, bgl.GL_TEXTURE_WRAP_T, bgl.GL_CLAMP_TO_EDGE)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

        # Bind shader that converts from scene linear to display space,
        # use the scene's color management settings.
        shader_program = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGetIntegerv(bgl.GL_CURRENT_PROGRAM, shader_program)

        # Generate vertex array
        self.vertex_array = bgl.Buffer(bgl.GL_INT, 1)
        bgl.glGenVertexArrays(1, self.vertex_array)
        bgl.glBindVertexArray(self.vertex_array[0])

        texturecoord_location = bgl.glGetAttribLocation(shader_program[0], "texCoord")
        position_location = bgl.glGetAttribLocation(shader_program[0], "pos")

        bgl.glEnableVertexAttribArray(texturecoord_location)
        bgl.glEnableVertexAttribArray(position_location)

        # Generate geometry buffers for drawing textured quad
        position = [0.0, 0.0, width, 0.0, width, height, 0.0, height]
        position = bgl.Buffer(bgl.GL_FLOAT, len(position), position)
        texcoord = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        texcoord = bgl.Buffer(bgl.GL_FLOAT, len(texcoord), texcoord)

        self.vertex_buffer = bgl.Buffer(bgl.GL_INT, 2)

        bgl.glGenBuffers(2, self.vertex_buffer)
        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[0])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, position, bgl.GL_STATIC_DRAW)
        bgl.glVertexAttribPointer(position_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[1])
        bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, texcoord, bgl.GL_STATIC_DRAW)
        bgl.glVertexAttribPointer(texturecoord_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

        bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)
        bgl.glBindVertexArray(0)

    def __del__(self):
        bgl.glDeleteBuffers(2, self.vertex_buffer)
        bgl.glDeleteVertexArrays(1, self.vertex_array)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)
        bgl.glDeleteTextures(1, self.texture)

    def draw(self):
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.texture[0])
        bgl.glBindVertexArray(self.vertex_array[0])
        bgl.glDrawArrays(bgl.GL_TRIANGLE_FAN, 0, 4)
        bgl.glBindVertexArray(0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)


# RenderEngines also need to tell UI Panels that they are compatible with.
# We recommend to enable all panels marked as BLENDER_RENDER, and then
# exclude any panels that are replaced by custom panels registered by the
# render engine, or that are not supported.
def get_panels():
    exclude_panels = {
        'VIEWLAYER_PT_filter',
        'VIEWLAYER_PT_layer_passes',
    }

    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
            if panel.__name__ not in exclude_panels:
                panels.append(panel)

    return panels


class TinaRenderProperties(bpy.types.PropertyGroup):
    render_samples: bpy.props.IntProperty(name='Render Samples', min=1, default=128)
    viewport_samples: bpy.props.IntProperty(name='Viewport Samples', min=1, default=32)
    start_pixel_size: bpy.props.IntProperty(name='Start Pixel Size', min=1, default=8, subtype='PIXEL')


def register():
    bpy.utils.register_class(TinaRenderProperties)

    #bpy.types.Light.tina_color = bpy.props.FloatVectorProperty(name='Color', subtype='COLOR', min=0, max=1, default=(1, 1, 1))
    #bpy.types.Light.tina_strength = bpy.props.FloatProperty(name='Strength', min=0, default=16, subtype='POWER')
    #bpy.types.Light.tina_radius = bpy.props.FloatProperty(name='Radius', min=0, default=0.1, subtype='DISTANCE')
    bpy.types.World.tina_color = bpy.props.FloatVectorProperty(name='Color', subtype='COLOR', min=0, max=1, default=(0.04, 0.04, 0.04))
    bpy.types.World.tina_strength = bpy.props.FloatProperty(name='Strength', min=0, default=1, subtype='POWER')
    bpy.types.Scene.tina_render = bpy.props.PointerProperty(name='tina', type=TinaRenderProperties)

    bpy.utils.register_class(TinaRenderEngine)
    bpy.utils.register_class(TinaLightPanel)
    bpy.utils.register_class(TinaWorldPanel)
    bpy.utils.register_class(TinaRenderPanel)

    for panel in get_panels():
        panel.COMPAT_ENGINES.add('TINA')

    global worker

    @mtworker.DaemonModule
    def worker():
        print('[TinaBlend] importing worker')
        from tina import worker
        print('[TinaBlend] importing worker done')
        return worker

    print('[TinaBlend] initializing worker')
    worker.init()
    print('[TinaBlend] initializing worker done')


def unregister():
    bpy.utils.unregister_class(TinaRenderEngine)
    bpy.utils.unregister_class(TinaLightPanel)
    bpy.utils.unregister_class(TinaWorldPanel)
    bpy.utils.unregister_class(TinaRenderPanel)

    for panel in get_panels():
        if 'TINA' in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove('TINA')

    #del bpy.types.Light.tina_color
    #del bpy.types.Light.tina_strength
    #del bpy.types.Light.tina_radius
    del bpy.types.World.tina_color
    del bpy.types.World.tina_strength
    del bpy.types.Scene.tina_render

    bpy.utils.unregister_class(TinaRenderProperties)
