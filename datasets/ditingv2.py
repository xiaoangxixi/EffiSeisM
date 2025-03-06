import pandas as pd
import json
import numpy as np
import h5py

from operator import itemgetter
from datasets.base import DatasetBase
import os
from ._factory import register_dataset

class DiTingv2(DatasetBase):
    """ DiTingv2 dataset"""

    _name: str = "DiTingv2"
    _channels = ["z", "n", "e"]
    _sampling_rate = 50
    _part_range = None
    def __init__(
        self,
        seed: int,
        mode: str,
        data_dir: str,
        shuffle: bool = True,
        data_split: bool = False,
        train_size: float = 0.8,
        val_size: float = 0.1,

        **kwargs
    ):
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size
        )



    def _load_meta_data(self, filename=None) -> pd.DataFrame:
        if filename is None:
            filename = "/data/lza/Ditingv2/CENC_DiTingv2_natural_earthquake.json"
        with open(filename, 'r') as file:
            data = json.load(file)

        meta_list = []
        for key, value in data.items():
            mag_type = value.get("magtype", "").lower()
            mag = value.get("mag", "")
            sg = value.get("Sg", 0)
            pg_dist = value.get("Pg_dist", 0)
            if mag_type == "ml" and mag != "   " and sg != 0 and pg_dist != 0 and pg_dist != '     ':
                # if se_mag !="":
                    meta_list.append({
                    "ev_id": key,
                    # "se_mag": float(value.get("se_mag", 0)),
                    "se_time": float(value.get("se_time", 0)),
                    # "sn_mag": float(value.get("sn_mag", 0)),
                    "sn_time": float(value.get("sn_time", 0)),
                    "Pg": float(value.get("Pg", 0)),
                    "Pg_res": value.get("Pg_res", ""),
                    "Pg_azi": value.get("Pg_azi", ""),
                    "Pg_dist": value.get("Pg_dist", ""),
                    "Sg": float(sg),
                    "Sg_res": value.get("Sg_res", ""),
                    "Sg_azi": value.get("Sg_azi", ""),
                    "Sg_dist": value.get("Sg_dist", ""),
                    "mag": float(mag),
                    "mag_type": mag_type,
                    "ev_type": value.get("evtype", ""),
                })

        meta_df = pd.DataFrame(meta_list)

        # # Filter the events with magnitude > 4
        # meta_df = meta_df[meta_df["mag"] > 3].copy()

        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)

        meta_df.reset_index(drop=True, inplace=True)

        if self._data_split and self._mode != "all":
            irange = {}
            irange["train"] = [0, int(self._train_size * meta_df.shape[0])]
            irange["val"] = [
                irange["train"][1],
                irange["train"][1] + int(self._val_size * meta_df.shape[0]),
            ]
            irange["test"] = [irange["val"][1], meta_df.shape[0]]

            r = irange[self._mode]
            meta_df = meta_df.iloc[r[0]: r[1], :]
            print(f"Data Split: {self._mode}: {r[0]}-{r[1]}")

        return meta_df

    def _load_event_data(self, idx: int) -> tuple:
        """Load event data

        Args:
            idx (int): Index.

        Raises:
            ValueError: Unknown 'mag_type'

        Returns:
            dict: Data of event.
            dict: Meta data.
        """

        target_event = self._meta_data.iloc[idx]
        key = target_event["ev_id"]

        # Assuming data is loaded from HDF5 file
        path = os.path.join(self._data_dir, 'CENC_DiTingv2_natural_earthquake.hdf5')
        with h5py.File(path, "r") as f:
            if key in f:
                data = np.array(f[key]).astype(np.float32).T
            else:
                raise KeyError(f"key {key} not found in HDF5 file")



        (
            setime,
            sntime,
            pg,
            pg_res,
            pg_azi,
            pg_dist,
            sg,
            sg_res,
            sg_azi,
            sg_dist,
            mag,
            mag_type,
        ) = itemgetter(
                "se_time",
                "sn_time",
                "Pg",
                "Pg_res",
                "Pg_azi",
                "Pg_dist",
                "Sg",
                "Sg_res",
                "Sg_azi",
                "Sg_dist",
                "mag",
                "mag_type",
        )(target_event)
        # Convert necessary variables to integers
        pg = int(pg)
        sg = int(sg)
        mag_type_lower = mag_type.lower()

        if isinstance(data, np.ndarray):
            if data.shape != (3, 8192):
                # 如果形状不符合要求，进行相应的转换
                if data.shape[1] > 8192:
                    # 如果数据的列数大于10000，截取前10000列
                    data = data[:, :8192]
                elif data.shape[1] < 8192:
                    # 计算需要添加的列数
                    padding_width = 8192 - data.shape[1]
                    # 在数据的后面添加 0
                    data = np.pad(data, ((0, 0), (0, padding_width)), mode='constant')
                # 检查转换后的形状是否为（3，10000）
                assert data.shape == (3, 8192)
        # Construct the event dictionary
        event = {
            "data": data,
            "ppks": [pg],
            "spks": [sg],
            "emg": [mag],
            "baz": [pg_azi],
            "dis": [pg_dist],
        }

        return event, target_event
#
@register_dataset
def ditingv2(**kwargs):
    dataset = DiTingv2( **kwargs)
    return dataset



# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     # 实例化DiTingv2类
#     dataset = DiTingv2(
#         seed=42,
#         mode="all",  # 包含所有数据
#         data_dir="/data/lza/Ditingv2",
#         shuffle=True,
#         data_split=False
#     )
#
#
#     meta_df = dataset._load_meta_data()
#
#     # Extract magnitude data and convert to float, filtering non-convertible values
#     magnitudes = pd.to_numeric(meta_df['Pg_dist'], errors='coerce').dropna().values
#
#     # 定义震中距的分箱，步长为20
#     bins = np.arange(0, max(magnitudes) + 20, 20)
#     hist, bin_edges = np.histogram(magnitudes, bins=bins)
#
#     # 生成渐变颜色，使用 colormap
#     log_hist = np.log10(hist + 1)  # 使用对数避免0的问题
#     norm = plt.Normalize(vmin=np.min(log_hist), vmax=np.max(log_hist))  # 归一化
#     cmap = plt.get_cmap('viridis')  # 使用渐变色调色板
#     colors = cmap(norm(log_hist))  # 生成每个柱子的颜色
#
#     # 计算每个柱子的宽度，确保宽度一致
#     bin_width = 20  # 根据分箱宽度设置柱子的宽度
#
#     # 绘制条形图，确保从0开始，且柱子居中
#     plt.figure(figsize=(10, 6),dpi=600)
#     bars = plt.bar(bin_edges[:-1] + (bin_width / 2), hist, width=bin_width, color=colors, edgecolor='black')
#
#     # 自定义图表
#     plt.yscale('log')  # 使用对数刻度
#     plt.xlabel('Distance (KM)', fontsize=14)
#     plt.ylabel('Number of samples', fontsize=14)
#     plt.xlim(0, 700)  # 设置 x 轴的范围
#     plt.ylim(1, 10 ** 6)  # 设置 y 轴范围
#     plt.xticks(np.arange(0, 701, 100), fontsize=12)
#     plt.yticks(fontsize=12)
#
#     # 添加渐变颜色条，指定 ax 参数
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])  # 仅用于颜色条
#     cbar = plt.colorbar(sm, ax=plt.gca())  # 指定当前坐标轴
#     cbar.set_label('Log(Number of samples)', fontsize=12)
#     # 更改颜色条的刻度数字大小
#     cbar.ax.tick_params(labelsize=14)  # 这里可以调整数字大小
#
#     # 显示图表
#     plt.tight_layout()
#     plt.show()
    # #  Define bins for magnitudes from 0.0 to 8.0 with a step of 0.1
    # bins = np.arange(0, 6.2, 0.2)
    # #
    # # Use np.histogram to count the occurrences within each bin
    # hist, bin_edges = np.histogram(magnitudes, bins=bins)
    # # 生成渐变颜色，使用 colormap
    # log_hist = np.log10(hist + 1)  # 使用对数避免 0 的问题
    # norm = plt.Normalize(vmin=np.min(log_hist), vmax=np.max(log_hist))  # 归一化
    # cmap = plt.get_cmap('plasma')  # 使用 plt.get_cmap 替代 cm.get_cmap
    # colors = cmap(norm(log_hist))  # 生成每个柱子的颜色
    #
    # # 计算每个柱子的宽度，确保宽度一致
    # bin_width = 0.2
    #
    # # 绘制条形图，确保从 0 开始，且柱子居中
    # plt.figure(figsize=(10, 6),dpi=600)
    # bars = plt.bar(bin_edges[:-1] + (bin_width / 2), hist, width=bin_width, color=colors, edgecolor='black')
    #
    # # 自定义图表
    # plt.yscale('log')  # 使用对数刻度
    # plt.xlabel('Magnitude (ML)', fontsize=14)
    # plt.ylabel('Number of samples', fontsize=14)
    # plt.xlim(0, 6)  # 设置 x 轴的范围
    # plt.ylim(1, 10 ** 6)  # 设置 y 轴范围
    # plt.xticks(np.arange(0, 6.5, 0.5), fontsize=12)
    # plt.yticks(fontsize=12)
    #
    # # 添加网格线
    # #plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #
    # # 添加渐变颜色条，指定 ax 参数
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # 仅用于颜色条
    # cbar = plt.colorbar(sm, ax=plt.gca())  # 指定当前坐标轴
    # cbar.set_label('Log(Number of samples)', fontsize=12)
    # # 更改颜色条的刻度数字大小
    # cbar.ax.tick_params(labelsize=14)  # 这里可以调整数字大小
    # # 显示图表
    # plt.tight_layout()
    # plt.show()



    # # Define bins with a step of 0.1
    # bins = np.arange(0, max(magnitudes) + 20, 20)
    # hist, bin_edges = np.histogram(magnitudes, bins=bins)
    # # Plot histogram
    # plt.figure(figsize=(10, 6),dpi=600)
    # plt.bar(bin_edges[1:], hist, width=20, edgecolor='black', color='blue')  # Adjusted to use bin_edges[1:]
    # plt.xlabel('Distance (KM)', fontsize=18)
    # plt.ylabel('Frequency', fontsize=18)  # Y-axis label
    # plt.xticks(np.arange(0, 701, 100))  # Set x-axis ticks from 0 to 700, step 100
    # plt.yscale('log')  # Set y-axis to log scale
    # plt.ylim(1, 10 ** 6)  # Set y-axis limits to 10^0 to 10^6
    # plt.xlim(0, 700)  # Set x-axis limits to 0 to 700
    # # # 设置坐标刻度字体大小
    # # plt.xticks(samples, fontsize=12)
    # plt.yticks(fontsize=12)
    #
    #
    # plt.show()

    # # Define bins with a step of 0.1
    # bins = np.arange(0, max(magnitudes) + 0.2, 0.2)
    # hist, bin_edges = np.histogram(magnitudes, bins=bins)
    # # Plot histogram
    # plt.figure(figsize=(10, 6),dpi=600)
    # plt.bar(bin_edges[1:], hist, width=0.2, edgecolor='black', color='orange')  # Adjusted to use bin_edges[1:]
    # plt.xlabel('Magnitude (ML)', fontsize=18)
    # plt.ylabel('Frequency', fontsize=18)  # Y-axis label
    # plt.xticks(np.arange(0, 8, 1))  # Set x-axis ticks from 0 to 700, step 100
    # plt.yscale('log')  # Set y-axis to log scale
    # plt.ylim(1, 10 ** 6)  # Set y-axis limits to 10^0 to 10^6
    # plt.xlim(0, 8)  # Set x-axis limits to 0 to 700
    # # # 设置坐标刻度字体大小
    # # plt.xticks(samples, fontsize=12)
    # plt.yticks(fontsize=12)
    #
    #
    # plt.show()



