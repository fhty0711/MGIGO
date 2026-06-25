bspline 轨迹的拼装办法：：：



1. bsplinetraj.py里面负责拼装整个轨迹。它接受目前物体的位置（x,y）和速度（vx，vy），然后
计算出前三个控制点P0 P1 和P2。
2. bsplinetraj.py 里面还需要写一个函数 splineguess，负责外推出P3 到PN。（时间全部按照\theta_i=0 后softmax分配）
这个将会用于Simple1.py 的初始猜测。
3. Simple1.py 里面，优化器操作的控制点是P3到PN。（我设置的控制点总数是N=10）.计算办法如下：
4. 从bsplinetraj.py 里面调用splineguess 得到初始猜测P3 到PN（否则求解器根本没法给好的结果）。
然后cost计算里面，调用bsplinetraj.py里面的拼装轨迹。做cost计算。

5. 重点重点重点！！！！！！！！！！！！！！！！！！！！！！！！！
求解器M=3 三个块！！！！！！！！！！！
分别装自由控制点和自由时间！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！