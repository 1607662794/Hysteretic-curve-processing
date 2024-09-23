'''该文件用于绘制滞回圈'''
import matplotlib.pyplot as plt


def plot_hysteresis_loop(force, displace, zero_number, reverse_disp, reverse_force, switch=True):
    if switch:  # switch控制是否开启多图显示
        num_loops = len(zero_number) - 1
        '''在多张图中显示'''
        for i in range(num_loops):
            start_idx = zero_number[i]
            end_idx = zero_number[i + 1]

            force_loop = force[start_idx:end_idx]
            displace_loop = displace[start_idx:end_idx]

            plt.figure()
            plt.plot(displace_loop, force_loop, label=f'Hysteresis Loop {i + 1}')
            # plt.xlabel('Displacement')
            # plt.ylabel('Force')
            # plt.title(f'Hysteresis Loop {i + 1}')
            plt.xlabel('位移(mm)')
            plt.ylabel('力(KN)')
            plt.title(f'Hysteresis Loop {i + 1}')
            plt.legend(fontsize='small')  # 设置图例字体大小为'small'
            plt.grid(True)
            if switch:
                plt.scatter(reverse_disp, reverse_force)
            plt.show()
            plt.savefig(f'E:\Code\Image regression\论文绘图\python绘图保存\滞回圈.png', dpi=600, bbox_inches='tight')
    else:
        '''在一张图中显示'''
        num_loops = len(zero_number)

        fig, ax = plt.subplots()

        for i in range(num_loops):
            start_idx = 0 if i == 0 else zero_number[i - 1]
            end_idx = zero_number[i]
            ax.plot(displace[start_idx:end_idx + 1], force[start_idx:end_idx + 1], label=f'Loop {i + 1}',
                    color=f'{(i / num_loops) / 1.2}')

        ax.set_xlabel('位移 (mm)', font={'family': 'SimSun', 'size': 22})
        ax.set_ylabel('荷载 (kN)', font={'family': 'SimSun', 'size': 22})
        # plt.title('滞回圈')
        plt.legend(prop={'size': 8.2}, loc=7)  # 设置图例字体大小为'small'
        xiloc = plt.MultipleLocator(2.5)  # Set a tick on each integer multiple of the *base* within the view interval.
        yiloc = plt.MultipleLocator(10)  # Set a tick on each integer multiple of the *base* within the view interval.
        ax.xaxis.set_minor_locator(xiloc)
        ax.yaxis.set_minor_locator(yiloc)
        ax.grid(color='lightgray', linestyle='-', linewidth=0.5, axis='both', which='both')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        # 刻度值字体大小设置（x轴和y轴同时设置）
        plt.tick_params(labelsize=18)
        # 移动轴的边缘来为刻度标注腾出空间
        plt.tight_layout()
        plt.savefig(f'E:\\研究生\\科研生活\\横向课题\\4.12-4.26冯老师教材\\绘图\\\滞回圈.png', dpi=600, bbox_inches='tight')
        plt.show()




if __name__ == "__main__":
    # 示例数据
    force = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2]
    displace = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    zero_number = [6, 11]

    # 绘制滞回圈图
    plot_hysteresis_loop(force, displace, zero_number)
