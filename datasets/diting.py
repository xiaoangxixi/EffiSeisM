from .base import DatasetBase
from typing import Optional, Tuple
import os
import pandas as pd
import numpy as np
from operator import itemgetter
import h5py
from utils import logger
from ._factory import register_dataset

"""
DiTing.

Reference:
    [1] Zhao, M., Xiao, Z., Chen, S., & Fang, L. (2023). 
        DiTing: A large-scale Chinese seismic benchmark dataset for artificial intelligence in seismology. 
        Earthquake Science, 36(2), 84-94.
        https://doi.org/10.1016/j.eqs.2022.01.022
"""


class DiTing(DatasetBase):
    """DiTing Dataset"""

    _name = "diting"
    _part_range = (0, 1)  # (inclusive,exclusive)
    _channels = ["z", "n", "e"]
    _sampling_rate = 50

    def __init__(
            self,
            seed: int,
            mode: str,
            data_dir: str,
            shuffle: bool = True,
            data_split: bool = True,
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
            val_size=val_size,
        )

    def _load_meta_data(self, filename=None) -> pd.DataFrame:
        start, end = self._part_range
        meta_df = pd.concat(
            [
                pd.read_csv(
                    os.path.join(self._data_dir, f"DiTing330km_part_{i}.csv"),
                    dtype={
                        "part": np.int64,
                        "key": str,
                        "ev_id": np.int64,
                        "evmag": str,
                        "mag_type": str,
                        "p_pick": np.int64,
                        "p_clarity": str,
                        "p_motion": str,
                        "s_pick": np.int64,
                        "net": str,
                        "sta_id": np.int64,
                        "dis": np.float32,
                        "st_mag": str,
                        "baz": str,
                        "Z_P_amplitude_snr": np.float32,
                        "Z_P_power_snr": np.float32,
                        "Z_S_amplitude_snr": np.float32,
                        "Z_S_power_snr": np.float32,
                        "N_P_amplitude_snr": np.float32,
                        "N_P_power_snr": np.float32,
                        "N_S_amplitude_snr": np.float32,
                        "N_S_power_snr": np.float32,
                        "E_P_amplitude_snr": np.float32,
                        "E_P_power_snr": np.float32,
                        "E_S_amplitude_snr": np.float32,
                        "E_S_power_snr": np.float32,
                        "P_residual": str,
                        "S_residual": str,
                    },
                    low_memory=False,
                    index_col=0,
                )
                for i in range(start, end)
            ]
        )
        meta_df["st_mag"] = meta_df["st_mag"].astype(float)

        for k in meta_df.columns:
            if meta_df[k].dtype in [object, np.object_, "object", "O"]:
                meta_df[k] = meta_df[k].str.replace(" ", "")

        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)

        meta_df.reset_index(drop=True, inplace=True)

        if self._data_split:
            irange = {}
            irange["train"] = [0, int(self._train_size * meta_df.shape[0])]
            irange["val"] = [
                irange["train"][1],
                irange["train"][1] + int(self._val_size * meta_df.shape[0]),
            ]
            irange["test"] = [irange["val"][1], meta_df.shape[0]]

            r = irange[self._mode]
            meta_df = meta_df.iloc[r[0]: r[1], :]
            logger.info(f"Data Split: {self._mode}: {r[0]}-{r[1]}")

        return meta_df

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        """Load evnet data

        Args:
            idx (int): Index.

        Raises:
            ValueError: Unknown 'mag_type'

        Returns:
            dict: Data of event.
            dict: Meta data.
        """

        target_event = self._meta_data.iloc[idx]
        part = target_event["part"]
        key = target_event["key"]
        key_correct = key.split(".")
        key = key_correct[0].rjust(6, "0") + "." + key_correct[1].ljust(4, "0")

        path = os.path.join(self._data_dir, f"DiTing330km_part_{part}.hdf5")
        with h5py.File(path, "r") as f:
            dataset = f.get("earthquake/" + str(key))
            data = np.array(dataset).astype(np.float32).T

        (
            ppk,
            spk,
            mag_type,
            evmag,
            stmag,
            motion,
            clarity,
            baz,
            dis,
            zpp_snr,
            nsp_snr,
            esp_snr,
        ) = itemgetter(
            "p_pick",
            "s_pick",
            "mag_type",
            "evmag",
            "st_mag",
            "p_motion",
            "p_clarity",
            "baz",
            "dis",
            "Z_P_power_snr",
            "N_S_power_snr",
            "E_S_power_snr",
        )(
            target_event
        )


        if pd.notnull(motion) and motion.lower() not in ["", "n"]:
            motion = {"u": 0, "c": 0, "r": 1, "d": 1}[motion.lower()]

        if pd.notnull(clarity):
            clarity = 0 if clarity.lower() == "i" else 1
        if pd.notnull(evmag) and evmag != '':
            # if evmag.strip() != "":
            #     evmag = float(evmag)
            # else:
            #     evmag = 0.0  # 或者提供一个默认值
            evmag=float(evmag)

        if pd.notnull(baz) and baz != '':
            # if baz.strip() != "":
            #     baz = float(baz)
            # else:
            #     baz = 0.0  # 或者提供一个默认值
            baz=float(baz)%360

        mag_type_lower = mag_type.lower()
        # To ml 
        if mag_type_lower == "ms":
            evmag = (evmag + 1.08) / 1.13
            stmag = (stmag + 1.08) / 1.13
        elif mag_type_lower == "mb":
            evmag = (1.17 * evmag + 0.67) / 1.13
            stmag = (1.17 * stmag + 0.67) / 1.13
        elif mag_type_lower == "ml":
            pass
        else:
            raise ValueError(f"Unknown 'mag_type' : '{mag_type}'")

        evmag = np.clip(evmag, 0, 8, dtype=np.float32)
        stmag = np.clip(stmag, 0, 8, dtype=np.float32)

        snr = np.array([zpp_snr, nsp_snr, esp_snr])

        event = {
            "data": data,
            "ppks": [ppk] if pd.notnull(ppk) else [],
            "spks": [spk] if pd.notnull(spk) else [],
            "emg": [evmag] if pd.notnull(evmag) else [],
            "smg": [stmag] if pd.notnull(stmag) else [],
            "pmp": [motion] if pd.notnull(motion) else [],
            "clr": [clarity] if pd.notnull(clarity) else [],
            "baz": [baz] if pd.notnull(baz) else [],
            "dis": [dis] if pd.notnull(dis) else [],
            "snr": snr,
        }

        return event, target_event.to_dict()


class DiTing_light(DiTing):
    _name = "diting_light"
    _part_range = None
    _channels = ["z", "n", "e"]
    _sampling_rate = 50

    def __init__(
            self,
            seed: int,
            mode: str,
            data_dir: str,
            shuffle: bool = True,
            data_split: bool = True,
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
            val_size=val_size,
        )

    def _load_meta_data(self, filename=f"DiTing330km_light.csv") -> pd.DataFrame:
        meta_df = pd.read_csv(
            os.path.join(self._data_dir, filename),
            dtype={
                "part": np.int64,
                "key": str,
                "ev_id": np.int64,
                "evmag": np.float32,
                "mag_type": str,
                "p_pick": np.int64,
                "p_clarity": str,
                "p_motion": str,
                "s_pick": np.int64,
                "net": str,
                "sta_id": np.int64,
                "dis": np.float32,
                "st_mag": np.float32,
                "baz": np.float32,
                "Z_P_amplitude_snr": np.float32,
                "Z_P_power_snr": np.float32,
                "Z_S_amplitude_snr": np.float32,
                "Z_S_power_snr": np.float32,
                "N_P_amplitude_snr": np.float32,
                "N_P_power_snr": np.float32,
                "N_S_amplitude_snr": np.float32,
                "N_S_power_snr": np.float32,
                "E_P_amplitude_snr": np.float32,
                "E_P_power_snr": np.float32,
                "E_S_amplitude_snr": np.float32,
                "E_S_power_snr": np.float32,
                "P_residual": np.float32,
                "S_residual": np.float32,
            },
            low_memory=False,
            index_col=0,
        )

        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)

        meta_df.reset_index(drop=True, inplace=True)

        if self._data_split:
            irange = {}
            irange["train"] = [0, int(self._train_size * meta_df.shape[0])]
            irange["val"] = [
                irange["train"][1],
                irange["train"][1] + int(self._val_size * meta_df.shape[0]),
            ]
            irange["test"] = [irange["val"][1], meta_df.shape[0]]

            r = irange[self._mode]
            meta_df = meta_df.iloc[r[0]: r[1], :]
            logger.info(f"Data Split: {self._mode}: {r[0]}-{r[1]}")

        return meta_df

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        """Load event data

        Args:
            idx (int): Index of target row.

        Returns:
            dict: Data of event.
            dict: Meta data.
        """
        return super()._load_event_data(idx=idx)


@register_dataset
def diting(**kwargs):
    dataset = DiTing(**kwargs)
    return dataset


@register_dataset
def diting_light(**kwargs):
    dataset = DiTing_light(**kwargs)
    return dataset

# if __name__ == '__main__':
#     from matplotlib import cm  # 用于渐变色
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd  # Ensure you import pandas
#
#     # Instantiate DiTing class
#     dataset = DiTing(
#         seed=42,
#         mode="all",  # Include all data
#         data_dir="/data/lza/Diting50hz/",
#         shuffle=True,
#         data_split=False
#     )
#
#     # Load metadata
#     meta_df = dataset._load_meta_data()
#
#     # Extract magnitude data and convert to float, filtering non-convertible values
#     magnitudes = pd.to_numeric(meta_df['dis'], errors='coerce').dropna().values


    # #  Define bins for magnitudes from 0.0 to 8.0 with a step of 0.1
    # bins = np.arange(0, 8, 0.2)
    # #
    # # Use np.histogram to count the occurrences within each bin
    # hist, bin_edges = np.histogram(magnitudes, bins=bins)

    # # 定义震中距的分箱，步长为20
    # bins = np.arange(0, max(magnitudes) + 10, 10)
    # hist, bin_edges = np.histogram(magnitudes, bins=bins)
    #
    # # 生成渐变颜色，使用 colormap
    # log_hist = np.log10(hist + 1)  # 使用对数避免0的问题
    # norm = plt.Normalize(vmin=np.min(log_hist), vmax=np.max(log_hist))  # 归一化
    # cmap = plt.get_cmap('viridis')  # 使用渐变色调色板
    # colors = cmap(norm(log_hist))  # 生成每个柱子的颜色
    #
    # # 计算每个柱子的宽度，确保宽度一致
    # bin_width = 10  # 根据分箱宽度设置柱子的宽度
    #
    # # 绘制条形图，确保从0开始，且柱子居中
    # plt.figure(figsize=(10, 6),dpi=600)
    # bars = plt.bar(bin_edges[:-1] + (bin_width / 2), hist, width=bin_width, color=colors, edgecolor='black')
    #
    # # 自定义图表
    # plt.yscale('log')  # 使用对数刻度
    # plt.xlabel('Distance (KM)', fontsize=14)
    # plt.ylabel('Number of samples', fontsize=14)
    # plt.xlim(0, 350)  # 设置 x 轴的范围
    # plt.ylim(1, 10 ** 6)  # 设置 y 轴范围
    # plt.xticks(np.arange(0, 350, 50), fontsize=12)
    # plt.yticks(fontsize=12)
    #
    # # 添加渐变颜色条，指定 ax 参数
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # 仅用于颜色条
    # cbar = plt.colorbar(sm, ax=plt.gca())  # 指定当前坐标轴
    #
    # cbar.set_label('Log(Number of samples)', fontsize=12)
    #
    # # 更改颜色条的刻度数字大小
    # cbar.ax.tick_params(labelsize=14)  # 这里可以调整数字大小
    #
    # # 显示图表
    # plt.tight_layout()
    # plt.show()
    #
    #
    #
    # # Plot the histogram
    # plt.figure(figsize=(10, 6))
    # plt.bar(bin_edges[:-1]+0.1, hist, width=0.2, color='orange', edgecolor='black')
    #
    # # Customize the plot
    # plt.yscale('linear')  # Make sure the y-axis is linear since we already took the log
    # plt.xlabel('Magnitude (ML)', fontsize=14)
    # plt.ylabel('Number of samples', fontsize=14)
    # plt.yscale('log')  # Set y-axis to log scale
    # plt.ylim(1, 10 ** 6)  # Set y-axis limits to 10^0 to 10^6
    # plt.xlim(0, 8)  # Set x-axis limits to 0 to 700
    # # # 设置坐标刻度字体大小
    # plt.yticks(fontsize=12)
    # # Adjust ticks and limits for better appearance
    # plt.xticks(np.arange(0, 8, 0.5),fontsize=12)
    #
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #
    # # Show the plot
    # plt.tight_layout()
    # plt.show()



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
    # plt.xlim(0, 7.8)  # 设置 x 轴的范围
    # plt.ylim(1, 10 ** 6)  # 设置 y 轴范围
    # plt.xticks(np.arange(0, 8.5, 0.5), fontsize=12)
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
    # bins = np.arange(0, max(magnitudes) + 10, 10)
    # hist, bin_edges = np.histogram(magnitudes, bins=bins)
    # # Plot histogram
    # plt.figure(figsize=(10, 6),dpi=600)
    # plt.bar(bin_edges[1:], hist, width=10, edgecolor='black', color='blue')  # Adjusted to use bin_edges[1:]
    # plt.xlabel('Distance (KM)', fontsize=18)
    # plt.ylabel('Frequency', fontsize=18)  # Y-axis label
    # plt.xticks(np.arange(0, 401, 100))  # Set x-axis ticks from 0 to 700, step 100
    # plt.yscale('log')  # Set y-axis to log scale
    # plt.ylim(1, 10 ** 6)  # Set y-axis limits to 10^0 to 10^6
    # plt.xlim(0, 400)  # Set x-axis limits to 0 to 700
    # # # 设置坐标刻度字体大小
    # # plt.xticks(samples, fontsize=12)
    # plt.yticks(fontsize=12)
    #
    #
    # plt.show()

    # Define bins with a step of 0.1
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
    # # Output maximum magnitude
    # max_magnitude = max(magnitudes)
    # print(f"最大震级是: {max_magnitude}")
