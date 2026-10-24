# M0 opponent baseline: RTX 4060 + KataGo (public-data approximation)

> 状态：**✅ 已闭合（近似口径，2026-10-24）**
>
> 对手机器不可访问。按项目决定，M0 第 ③ 项使用公开硬件基准做量级估算，
> 不声称这是对手实测。该数字只用于赌约条款和教师预算规划，不能用于正式 A/B。

## 公开基准

来源：KataGo GitHub issue
[#852](https://github.com/lightvector/KataGo/issues/852)。

| 项目 | 值 |
|---|---|
| 主机 | i5-12400F / RTX 4060 Ti / 16 GB |
| 网络 | KataGo 40b |
| 配置 | `default_gtp.cfg` |
| OpenCL | **1484.78 visits/s** / 1239.38 evals/s |
| TensorRT | **1956.56 visits/s** / 1780.69 evals/s |
| CUDA | 253.89 visits/s（该样本异常，不纳入估算） |

## RTX 4060 近似

桌面 RTX 4060 Ti → RTX 4060 的主要缩放因子：

- CUDA core：4352 → 3072（70.6%）
- 显存带宽：288 → 272 GB/s（94.4%）
- 实际 KataGo 吞吐取 **70%–95%** 的 Ti 实测值：
  - OpenCL：约 **1040–1410 visits/s**
  - TensorRT：约 **1370–1860 visits/s**

**规划值：约 1200 visits/s（OpenCL，低置信度）。**
不确定性约 ±40%；若对手使用 Laptop 4060、功耗受限或不同网络，实际值可能明显更低。

## 后续替换规则

只要拿到对手使用的 KataGo 版本、配置、网络和 `katago benchmark` 输出，就用实测值替换本节，
并把 M0 的依据从“公开近似”升级为“对手实测”。在拿到之前，以上近似已经满足 M0 的数量级门禁。
