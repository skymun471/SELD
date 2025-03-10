# import numpy as np
#
# # 읽고자 하는 npy 파일의 경로를 지정합니다.
# npy_file_path = '/home/milab11/Desktop/mhn_ws/SELD/seld-dcase2022/seld_feat_label/mic_dev_adpit_label/fold3_room4_mix001.npy'
#
# # npy 파일 로드
# data = np.load(npy_file_path)
#
# print(data.shape) # (1290, 6, 4, 13)
# print("label",data[])
# # 로드한 데이터 출력
# # print("Loaded data from {}: \n{}".format(npy_file_path, data))

# import numpy as np
#
# # npy 파일 경로 지정
# npy_file_path = '/home/milab11/Desktop/mhn_ws/SELD/seld-dcase2022/seld_feat_label/mic_dev_adpit_label/fold3_room4_mix001.npy'
#
# # npy 파일 로드
# data = np.load(npy_file_path)
#
# # 전체 데이터의 shape 출력
# print("전체 라벨 데이터 shape:", data.shape)  # 예: (1290, 6, 4, 13)
#
# # 예제 1: 특정 시간 프레임(예: 100번째 프레임)의 모든 트랙에 대한 라벨 정보를 출력
# time_index = 100
# print("\nTime frame {}의 라벨 정보:".format(time_index))
# for track_idx in range(data.shape[1]):  # data.shape[1]는 트랙(또는 소스)의 개수, 여기서는 6
#     label_info = data[time_index, track_idx]  # shape: (4, 13)
#     print("\nTrack {}:".format(track_idx))
#     print(label_info)
#
# # 예제 2: 첫 번째 트랙의 라벨 정보를 시간에 따라 확인
# track_index = 0
# print("\nTrack {}의 라벨 정보를 시간에 따라 출력:".format(track_index))
# for t in range(data.shape[0]):  # data.shape[0]는 시간 프레임 개수, 여기서는 1290
#     # 각 시간 프레임마다 (4, 13) 배열이 있는데, 여기서는 간단히 첫 DOA 축(예: x축)의 값을 기준으로
#     # 활성화된 클래스(예: 값이 특정 threshold 이상인)를 확인하는 예시를 보여줍니다.
#     label_frame = data[t, track_index]  # shape: (4, 13)
#     # 예시: 첫번째 DOA 축의 값으로 활성 클래스 판단 (threshold 0.5)
#     active_classes = np.where(label_frame[0] > 0.5)[0]  # 첫 번째 축의 값이 0.5 이상인 인덱스
#     print("Frame {}: 활성 클래스 인덱스: {}".format(t, active_classes))
#
#     # 만약 첫 10 프레임만 확인하고 싶다면:
#     if t >= 3000:
#         break

import numpy as np

# npy 파일 경로 지정
npy_file_path = '/home/milab11/Desktop/mhn_ws/SELD/seld-dcase2022/seld_feat_label/mic_dev_adpit_label/fold3_room4_mix001.npy'

# npy 파일 로드
data = np.load(npy_file_path)

# 전체 데이터의 shape 출력 (예: (1290, 6, 4, 13))
print("전체 라벨 데이터 shape:", data.shape)

# 예제: 첫 10 프레임의 모든 GT 값을 출력
num_frames_to_print = 1200  # 원하는 프레임 개수로 조절 가능
for t in range(num_frames_to_print):
    print("\nFrame {}의 전체 GT 값:".format(t))
    # 각 프레임마다 6개의 트랙이 있으므로 각 트랙에 대해 (4, 13) 배열 출력
    for track_idx in range(data.shape[1]):
        print("  Track {}:".format(track_idx))
        print(data[t, track_idx])
    print("-" * 50)

