from tina.common import *
from tina.tools import matrix
from base64 import b64decode
from gltflib import *


def readgltf(path):
    root = GLTF.load(path)
    model = root.model
    buffers = []
    bufferViews = []
    accessors = []
    primitives = []


    def load_uri(uri):
        if uri.startswith('data:'):
            index = uri.index('base64,')
            data = uri[index + len('base64,'):]
            data = b64decode(data.encode('ascii'))
        else:
            with open(uri, 'rb') as f:
                data = f.read()
        return data


    for i, buffer in enumerate(model.buffers):
        print('loading buffer', i)
        buffers.append(load_uri(buffer.uri))


    def get_node_local_matrix(node):
        mat = matrix.identity()
        if node.scale is not None:
            mat = matrix.scale(node.scale) @ mat
        if node.rotation is not None:
            mat = matrix.quaternion(node.rotation) @ mat
        if node.translation is not None:
            mat = matrix.translate(node.translation) @ mat
        return mat


    def get_bufferView_bytes(bufferView):
        offset = bufferView.byteOffset
        length = bufferView.byteLength
        buffer = buffers[bufferView.buffer]
        buffer = buffer[offset:offset + length]
        return buffer


    for bufferView in model.bufferViews:
        bufferViews.append(get_bufferView_bytes(bufferView))


    def get_accessor_buffer(accessor):
        component_types = 'bBhHiIf234d'
        vector_types = {
            'SCALAR': '',
            'VEC2': '2',
            'VEC3': '3',
            'VEC4': '4',
        }

        buffer = bufferViews[accessor.bufferView]

        dtype = component_types[accessor.componentType - 0x1400]
        dtype = vector_types[accessor.type] + dtype
        count = accessor.count

        array = np.frombuffer(buffer, dtype=dtype, count=count)
        return array


    for accessor in model.accessors:
        accessors.append(get_accessor_buffer(accessor))


    def process_material(material):
        pbr = material.pbrMetallicRoughness
        basecolor = pbr.baseColorFactor[:3]
        basecolorTex = pbr.baseColorTexture
        metallic = pbr.metallicFactor
        roughness = pbr.roughnessFactor
        metallicRoughnessTex = pbr.metallicTexture


    for material in model.materials:
        materials.append(process_material(material))


    def process_primitive(prim, world):
        indices = prim.indices
        material = prim.material
        position = prim.attributes.POSITION
        normal = prim.attributes.NORMAL
        texcoord = prim.attributes.TEXCOORD_0
        if position is not None:
            position = accessors[position]
        if normal is not None:
            normal = accessors[normal]
        if texcoord is not None:
            texcoord = accessors[texcoord]
        if indices is not None:
            indices = accessors[indices]
        if material is not None:
            mtlid = materials[material]
        primitives.append((position, normal, texcoord, world, indices, mtlid))


    def process_mesh(mesh, world):
        print('processing mesh', mesh.name)
        for prim in mesh.primitives:
            process_primitive(prim, world)


    def _process_node(node, world):
        if node.mesh is not None:
            process_mesh(model.meshes[node.mesh], world)


    def process_node(node, world=None):
        if world is None:
            world = matrix.identity()
        print('processing node', node.name)
        local = get_node_local_matrix(node)
        world = world @ local
        _process_node(node, world)
        if node.children is not None:
            for child in node.children:
                process_node(model.nodes[child], world)


    def process_scene(scene):
        for node in scene.nodes:
            process_node(model.nodes[node])


    process_scene(model.scenes[model.scene])


    def np34(x, w):
        return np.concatenate([x, np.full((x.shape[0], 1), w)], axis=1)


    def np43(x):
        return x[:, :3] / np.repeat(x[:, 3, None], 3, axis=1)


    def npnmlz(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)


    def transform_primitive(p, n, t, w, f):
        assert w is not None
        assert p is not None
        assert n is not None
        assert f is not None

        if t is None:
            t = np.zeros((p.shape[0], 2))

        p = p.astype(np.float64)[f]
        n = n.astype(np.float64)[f]
        t = t.astype(np.float64)[f]
        w = w.astype(np.float64)

        w = w.transpose()
        p = np43(np34(p, 1) @ w)
        n = npnmlz((np34(n, 0) @ w)[:, :3])

        a = np.concatenate([p, n, t], axis=1)
        return a


    arrays = []
    mtlids = []
    for p, n, t, w, f in primitives:
        a, m = transform_primitive(p, n, t, w, f)
        arrays.append(a)
        mtlids.append(m)
    assert len(arrays) and len(mtlids)

    arr = np.concatenate(arrays, axis=0)
    mtlids = np.concatenate(mtlids, axis=0)

    return arr, mtlids


if __name__ == '__main__':
    arr, mtlids = readgltf('assets/luxball.gltf')
    print(arr, mtlids)
