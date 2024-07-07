import torch
from monoscene.models.monoscene import MonoScene

torch.manual_seed(84)
torch.cuda.manual_seed_all(84)


path = 'trained_models/monoscene_nyu.ckpt'
model = MonoScene.load_from_checkpoint(
        path,
        feature=200,
        project_scale=1,
        fp_loss=True,
        full_scene_size=(60, 36, 60),
        voxeldepth=False,
        voxeldepth_res = [8],
    )
model = model.net_rgb.encoder
model_state_dict = model.state_dict()
from prettytable import PrettyTable
tb = PrettyTable(['Layers', 'Sizes', 'devices'])
for k, v in model_state_dict.items():
    # state_str += k + '\t' + str(v.shape) + '\n'
    tb.add_row([k, str(v.shape), str(v.device)])
print(tb)


input = torch.randn(1, 3, 480, 640)
device_all = ['cpu', 'cuda']
output_all = []

for device in device_all:

    if device == 'cpu':
        pass
    elif device == 'cuda':
        model = model.cuda()
        input = input.cuda()
    model.eval()
    with torch.no_grad():
        output = model(input)
    output_all.append(output[6])
    # print(output[6], 'output')
    print(input.device, output[0].device)
diff = torch.abs(output_all[1].cpu()-output_all[0])
print(output_all[1])
print(diff.min(), diff.max())

# '''Save Model Parameters'''
#     model.eval()
#     torch.save(model.net_rgb.encoder.original_model.state_dict(), '/home/hongxiao.yu/mmdetection3d-occ/encoder_pl.pt')  # save encoder params
#     torch.save(model.net_rgb.decoder.state_dict(), '/home/hongxiao.yu/mmdetection3d-occ/decoder_pl.pt')
#     torch.save(model.state_dict(), '/home/hongxiao.yu/mmdetection3d-occ/monoscene_pl.pt')  # save all model params
#     # for k, v in model.net_rgb.encoder.state_dict().items():
#     #     if 'original_model.blocks.1.1.conv_pw.weight' in k:
#     #         print(k, v, v.dtype)
#     # import torch.nn as nn
#     # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     # model.net_rgb.encoder.v

#     torch.manual_seed(84)
#     input = torch.randn(2, 3, 480, 640).cpu()
#     model = model.cpu()
#     model.eval()
#     output = model.net_rgb.encoder(input)
#     print(output[-1], 'output')