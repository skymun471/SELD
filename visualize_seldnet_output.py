#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#
import numpy as np
import os
import sys
import cls_data_generator
import seldnet_model
import parameters
import torch
from IPython import embed
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
plot.rcParams.update({'font.size': 22})

# multi-accdoa를 위한 함수 추가
# ==============================================================================
from cls_compute_seld_results import reshape_3Dto2D
# ==============================================================================


def main(argv):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    print('\nLoading the best model and predicting results on the testing split')
    print('\tLoading testing dataset:')

    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=3, shuffle=False, is_eval=True if params['mode']=='eval' else False
    )

    data_in, data_out = data_gen_test.get_data_sizes()
    print(f"feat_shape:{data_in}, label_shape:{data_out}")
    dump_figures = True

    # CHOOSE THE MODEL WHOSE OUTPUT YOU WANT TO VISUALIZE 
    checkpoint_name = "/home/milab11/Desktop/mhn_ws/SELD/seld-dcase2022/models/4_1_dev_split0_accdoa_mic_gcc_model.h5"
    model = seldnet_model.CRNN(data_in, data_out, params)
    model.eval()
    model.load_state_dict(torch.load(checkpoint_name, map_location=torch.device('cpu')))
    model = model.to(device)

    if dump_figures:
        dump_folder = os.path.join('dump_dir', os.path.basename(checkpoint_name).split('.')[0])
        os.makedirs(dump_folder, exist_ok=True)


    with torch.no_grad():
        file_cnt = 0
        for data, target in data_gen_test.generate():

            # # 원하는 파일의 인덱스를 예: 5로 설정
            # if file_cnt != 8:
            #     file_cnt += 1
            #     continue

            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
            output = model(data)
            print(f"output shape 1:{output.shape}")

            # (batch, sequence, max_nb_doas*3) to (batch, sequence, 3, max_nb_doas)
            # ========================================
            max_nb_doas = output.shape[2]//3
            print(f"max_nb_doas : {max_nb_doas}")

            # multi accdoa?
            # max_nb_doas = 39
            # ========================================

            output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
            print("Target output shape:", target.shape, output.shape)
            target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)
            print("Target output shape:", target.shape, output.shape)
            # get pair-wise distance matrix between predicted and reference.
            output, target = output.view(-1, output.shape[-2], output.shape[-1]), target.view(-1, target.shape[-2], target.shape[-1])

            output = output.cpu().detach().numpy()
            target = target.cpu().detach().numpy()



            use_activity_detector = False
            if use_activity_detector:
                activity = (torch.sigmoid(activity_out).cpu().detach().numpy() >0.5)
            mel_spec = data[0][0].cpu()
            foa_iv = data[0][-1].cpu()
            target[target > 1] =0

            plot.figure(figsize=(20,10))
            plot.subplot(321), plot.imshow(torch.transpose(mel_spec, -1, -2))
            plot.subplot(322), plot.imshow(torch.transpose(foa_iv, -1, -2))

            plot.subplot(323), plot.plot(target[:params['label_sequence_length'], 0, 0], 'r', lw=2)
            plot.subplot(323), plot.plot(target[:params['label_sequence_length'], 0, 1], 'g', lw=2)
            plot.subplot(323), plot.plot(target[:params['label_sequence_length'], 0, 2], 'b', lw=2)
            plot.grid()
            plot.ylim([-1.1, 1.1])

            plot.subplot(324), plot.plot(target[:params['label_sequence_length'], 1, 0], 'r', lw=2)
            plot.subplot(324), plot.plot(target[:params['label_sequence_length'], 1, 1], 'g', lw=2)
            plot.subplot(324), plot.plot(target[:params['label_sequence_length'], 1, 2], 'b', lw=2)
            plot.grid()
            plot.ylim([-1.1, 1.1])
            if use_activity_detector:
                output[:, 0, 0:3] = activity[:, 0][:, np.newaxis]*output[:, 0, 0:3]
                output[:, 1, 0:3] = activity[:, 1][:, np.newaxis]*output[:, 1, 0:3]

            plot.subplot(325), plot.plot(output[:params['label_sequence_length'], 0, 0], 'r', lw=2)
            plot.subplot(325), plot.plot(output[:params['label_sequence_length'], 0, 1], 'g', lw=2)
            plot.subplot(325), plot.plot(output[:params['label_sequence_length'], 0, 2], 'b', lw=2)
            plot.grid()
            plot.ylim([-1.1, 1.1])

            plot.subplot(326), plot.plot(output[:params['label_sequence_length'], 1, 0], 'r', lw=2)
            plot.subplot(326), plot.plot(output[:params['label_sequence_length'], 1, 1], 'g', lw=2)
            plot.subplot(326), plot.plot(output[:params['label_sequence_length'], 1, 2], 'b', lw=2)
            plot.grid()
            plot.ylim([-1.1, 1.1])

            if dump_figures:
                fig_name = '{}'.format(os.path.join(dump_folder, '{}.png'.format(file_cnt)))
                print('saving figure : {}'.format(fig_name))
                plot.savefig(fig_name, dpi=100)
                plot.close()
                file_cnt += 1
            else:
                plot.show()
            if file_cnt>2:
                break

    # multi-accdoa 지원 코드 추가 미완성
    # ==============================================================================
    # with torch.no_grad():
    #     file_cnt = 0
    #     for data, target in data_gen_test.generate():
    #         data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
    #         output = model(data)
    #         print(f"Original output shape: {output.shape}")
    #
    #         if params['multi_accdoa']:
    #             # multi_accdoa 후처리: 모델 출력은 Multi-ACCDOA 형식으로 가정 (예: [batch, seq, 6, 4, num_classes])
    #             # get_multi_accdoa_labels() 함수를 통해 SED와 DOA를 분리하고, reshape_3Dto2D()로 2D 형태로 변환
    #             output_np = output.detach().cpu().numpy()
    #             sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = \
    #                 get_multi_accdoa_labels(output_np, params['unique_classes'])
    #             sed_pred0 = reshape_3Dto2D(sed_pred0)
    #             doa_pred0 = reshape_3Dto2D(doa_pred0)
    #             sed_pred1 = reshape_3Dto2D(sed_pred1)
    #             doa_pred1 = reshape_3Dto2D(doa_pred1)
    #             sed_pred2 = reshape_3Dto2D(sed_pred2)
    #             doa_pred2 = reshape_3Dto2D(doa_pred2)
    #             # 예시로, 여기서는 여러 결과 중 첫 번째 SED와 DOA를 시각화에 사용한다고 가정합니다.
    #             output_processed = (sed_pred0, doa_pred0)
    #             print(f"Multi-ACCDOA processed output shapes: SED: {sed_pred0.shape}, DOA: {doa_pred0.shape}")
    #         else:
    #             # 일반 ACCDOA 처리: 출력 shape가 [batch, seq, max_nb_doas*3]
    #             max_nb_doas = output.shape[2] // 3
    #             print(f"max_nb_doas : {max_nb_doas}")
    #             output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
    #             # flatten to (batch*seq, DOA_dim)
    #             output_processed = output.view(-1, output.shape[-2], output.shape[-1]).detach().cpu().numpy()
    #             print(f"Processed output shape: {output_processed.shape}")
    #
    #         # target 처리: 학습 시 task_id=6로 생성한 target은 [batch, seq, 6, 4, num_classes] 형식이어야 함.
    #         # 만약 multi_accdoa로 학습했다면, target은 이미 올바른 형식이므로 그대로 사용하거나
    #         # 별도의 후처리가 필요하면 같은 함수를 적용합니다.
    #         if params['multi_accdoa']:
    #             target_processed = target.detach().cpu().numpy()
    #             print(f"Multi-ACCDOA target shape: {target_processed.shape}")
    #         else:
    #             target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)
    #             target_processed = target.view(-1, target.shape[-2], target.shape[-1]).detach().cpu().numpy()
    #             print(f"Processed target shape: {target_processed.shape}")
    #
    #         # 이후 output_processed와 target_processed를 이용하여 시각화를 진행합니다.
    #         # 예시: mel_spec, foa_iv 등은 data에서 추출하는 방식은 그대로 사용.
    #         mel_spec = data[0][0].cpu()
    #         foa_iv = data[0][-1].cpu()
    #         target_processed[target_processed > 1] = 0
    #
    #         plot.figure(figsize=(20, 10))
    #         plot.subplot(321), plot.imshow(torch.transpose(mel_spec, -1, -2))
    #         plot.subplot(322), plot.imshow(torch.transpose(foa_iv, -1, -2))
    #
    #         # 예시로 target_processed의 첫 채널을 시각화 (여기서 적절한 채널 선택은 상황에 따라 조정)
    #         plot.subplot(323), plot.plot(target_processed[:params['label_sequence_length'], 0, 0], 'r', lw=2)
    #         plot.subplot(323), plot.plot(target_processed[:params['label_sequence_length'], 0, 1], 'g', lw=2)
    #         plot.subplot(323), plot.plot(target_processed[:params['label_sequence_length'], 0, 2], 'b', lw=2)
    #         plot.grid()
    #         plot.ylim([-1.1, 1.1])
    #
    #         plot.subplot(324), plot.plot(target_processed[:params['label_sequence_length'], 1, 0], 'r', lw=2)
    #         plot.subplot(324), plot.plot(target_processed[:params['label_sequence_length'], 1, 1], 'g', lw=2)
    #         plot.subplot(324), plot.plot(target_processed[:params['label_sequence_length'], 1, 2], 'b', lw=2)
    #         plot.grid()
    #         plot.ylim([-1.1, 1.1])
    #
    #         if params['multi_accdoa']:
    #             # multi_accdoa의 경우, output_processed는 tuple (sed_pred0, doa_pred0) 등으로 구성되어 있습니다.
    #             # 여기서는 예시로 sed_pred0와 doa_pred0를 시각화합니다.
    #             sed_to_plot, doa_to_plot = output_processed  # 예시
    #             plot.subplot(325), plot.plot(sed_to_plot[:params['label_sequence_length'], 0, 0], 'r', lw=2)
    #             plot.subplot(325), plot.plot(sed_to_plot[:params['label_sequence_length'], 0, 1], 'g', lw=2)
    #             plot.subplot(325), plot.plot(sed_to_plot[:params['label_sequence_length'], 0, 2], 'b', lw=2)
    #             plot.grid()
    #             plot.ylim([-1.1, 1.1])
    #             plot.subplot(326), plot.plot(doa_to_plot[:params['label_sequence_length'], 1, 0], 'r', lw=2)
    #             plot.subplot(326), plot.plot(doa_to_plot[:params['label_sequence_length'], 1, 1], 'g', lw=2)
    #             plot.subplot(326), plot.plot(doa_to_plot[:params['label_sequence_length'], 1, 2], 'b', lw=2)
    #             plot.grid()
    #             plot.ylim([-1.1, 1.1])
    #         else:
    #             # 일반 ACCDOA의 경우, output_processed는 numpy array로 이미 reshape되어 있습니다.
    #             plot.subplot(325), plot.plot(output_processed[:params['label_sequence_length'], 0, 0], 'r', lw=2)
    #             plot.subplot(325), plot.plot(output_processed[:params['label_sequence_length'], 0, 1], 'g', lw=2)
    #             plot.subplot(325), plot.plot(output_processed[:params['label_sequence_length'], 0, 2], 'b', lw=2)
    #             plot.grid()
    #             plot.ylim([-1.1, 1.1])
    #             plot.subplot(326), plot.plot(output_processed[:params['label_sequence_length'], 1, 0], 'r', lw=2)
    #             plot.subplot(326), plot.plot(output_processed[:params['label_sequence_length'], 1, 1], 'g', lw=2)
    #             plot.subplot(326), plot.plot(output_processed[:params['label_sequence_length'], 1, 2], 'b', lw=2)
    #             plot.grid()
    #             plot.ylim([-1.1, 1.1])
    #
    #         if dump_figures:
    #             fig_name = os.path.join(dump_folder, f'{file_cnt}.png')
    #             print('saving figure : {}'.format(fig_name))
    #             plot.savefig(fig_name, dpi=100)
    #             plot.close()
    #             file_cnt += 1
    #         else:
    #             plot.show()
    #         if file_cnt > 2:
    #             break
    # ==============================================================================

# multi-accdoa 시각화를 위한 함수 추가
# ==============================================================================
# def get_multi_accdoa_labels(accdoa_in, nb_classes):
#     """
#     Args:
#         accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
#         nb_classes: scalar
#     Return:
#         sedX:       [batch_size, frames, num_class=12]
#         doaX:       [batch_size, frames, num_axis*num_class=3*12]
#     """
#     x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
#     sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
#     doa0 = accdoa_in[:, :, :3*nb_classes]
#
#     x1, y1, z1 = accdoa_in[:, :, 3*nb_classes:4*nb_classes], accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes]
#     sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
#     doa1 = accdoa_in[:, :, 3*nb_classes: 6*nb_classes]
#
#     x2, y2, z2 = accdoa_in[:, :, 6*nb_classes:7*nb_classes], accdoa_in[:, :, 7*nb_classes:8*nb_classes], accdoa_in[:, :, 8*nb_classes:]
#     sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
#     doa2 = accdoa_in[:, :, 6*nb_classes:]
#
#     return sed0, doa0, sed1, doa1, sed2, doa2
# ==============================================================================


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)


