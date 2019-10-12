import pickle
import PIL.Image
import numpy as np
import dnnlib.tflib as tflib
from util.generator_model import Generator

def read_feature(file_name):
    file = open(file_name, mode='r')
    # 使用readlines() 读取所有行的数据，会返回一个列表，列表中存放的数据就是每一行的内容
    contents = file.readlines()
    # 准备一个列表，用来存放取出来的数据
    code = np.zeros((512, ))
    # for循环遍历列表，去除每一行读取到的内容
    for i in range(512):
        name = contents[i]
        name = name.strip('\n')
        code[i] = name
    code = np.float32(code)
    file.close()
    return code

def generate_image(latent_vector, generator):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img

def move_latent_and_save(latent_vector, direction, coeffs, generator):
    '''latent_vector是人脸潜编码，direction是人脸调整方向，coeffs是变化步幅的向量，generator是生成器'''
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        result = generate_image(new_latent_vector, generator)
        result.save('results/'+str(i).zfill(3)+'.png')


def main():

    # 在这儿选择生成器
    tflib.init_tf()
    with open('model/generator_yellow.pkl', "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

    # 在这儿选择人物的潜码，注意要与生成器相匹配。潜码来自生成目录下有个generate_codes文件夹里的txt文件。
    face_latent = read_feature('input_latent/0001.txt')
    stack_latents = np.stack(face_latent for _ in range(1))
    face_dlatent = Gs_network.components.mapping.run(stack_latents, None)

    # 在这儿选择调整的方向，共有23种调整方式，它们的名称与分别对应的功能如下所示。
    '''
        age.npy - 调整年龄
        angle_horizontal.npy - 在左右方向上调整人脸角度
        angle_vertical.npy - 在上下方向上调整人脸角度
        beauty.npy - 调整颜值
        camera_rotation.npy - 调整摄像机角度
        emotion_angry.npy - 调整此项可增添/减弱一些生气的情绪（调整步幅建议缩小）
        emotion_disgust.npy - 调整此项可增添/减弱一些厌恶的情绪（调整步幅建议缩小）
        emotion_easy.npy - 调整此项可增添/减弱一些平静的情绪（调整步幅建议缩小）
        emotion_fear.npy - 调整此项可增添/减弱一些害怕的情绪（调整步幅建议缩小）
        emotion_happy.npy - 调整此项可增添/减弱一些开心的情绪（调整步幅建议缩小）
        emotion_sad.npy - 调整此项可增添/减弱一些伤心的情绪（调整步幅建议缩小）
        emotion_surprise.npy - 调整此项可增添/减弱一些惊讶的情绪（调整步幅建议缩小）
        exposure.npy - 调整光照
        eyes_open.npy - 调整眼睛的闭合程度
        face_shape.npy - 调整脸型
        gender.npy - 调整性别
        glasses.npy - 调整是否戴眼镜
        height.npy - 调整脸的高度
        race_black.npy - 调整此项可接近/远离向黑种人变化
        race_white.npy - 调整此项可接近/远离向白种人变化
        race_yellow.npy - 调整此项可接近/远离向黄种人变化
        smile.npy - 调整笑容
        width.npy - 调整脸的宽度
    '''
    direction = np.load('latent_directions/smile.npy')  # 从上面的编辑向量中选择一个

    # 在这儿选择调整的大小，向量里面的值表示调整幅度，可以自行编辑，对于每个值都会生成一张图片并保存。
    coeffs = [-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.]

    # 开始调整并保存图片
    move_latent_and_save(face_dlatent, direction, coeffs, generator)


if __name__ == "__main__":
    main()