'''该文件用于绘制滞回圈'''
import matplotlib.pyplot as plt


def plot_hysteresis_loop(force, displace, zero_number,reverse_disp,reverse_force, switch=True):
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
    else:
        '''在一张图中显示'''
        num_loops = len(zero_number)

        for i in range(num_loops):
            start_idx = 0 if i == 0 else zero_number[i - 1]
            end_idx = zero_number[i]

            plt.plot(displace[start_idx:end_idx + 1], force[start_idx:end_idx + 1], label=f'Loop {i + 1}')

        # plt.xlabel('Displacement')
        # plt.ylabel('Force')
        # plt.title('Hysteresis Loop')
        plt.xlabel('位移(mm)')
        plt.ylabel('力(KN)')
        plt.title('滞回圈')
        plt.legend(fontsize='small')  # 设置图例字体大小为'small'
        plt.grid(True)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.show()


if __name__ == "__main__":
    # 示例数据
    force = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2]
    displace = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    zero_number = [6, 11]

    # 绘制滞回圈图
    plot_hysteresis_loop(force, displace, zero_number)
