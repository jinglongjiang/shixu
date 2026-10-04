# Latent Temporal Information Audit

日期：2026-10-04。范围：本地 shixu 的原生5人 circle/square 环境；离线信息恢复和小规模动作后果诊断。**没有训练导航新模型，没有启动KDA V9，没有修改环境、奖励或已有checkpoint。**

## 1. 当前结论

1. **历史有可测的信息增量，但不是跨场景稳定的长历史优势。** 在circle留出轨迹上，24帧探针优于当前帧探针；简单历史统计也能获得相近或更好的恢复结果。circle训练后的长历史探针在square上明显退化，CV很强。
2. **隐藏目标具有局部决策价值。** 在8个近身相遇状态中，补充真实目标相对当前帧估计能减少部分剩余到达时间；只补ORCA平滑状态没有稳定改善。这是离线oracle结果，不是可部署新方法的成绩。
3. **circle里的这部分收益可以非常简单。** 保存合法首次看见的位置，利用circle的对径目标规则推算目标，在4个近身circle状态、两种continuation下均达到真值参考的最高回报。只需两个位置数，不需要复杂关联记忆。
4. **没有得到KDA研发重新启动的证据。** 也没有证明整个时序方向没有价值。下一步应区分可泛化运动隐状态与生成器特有的起点—目标规则，而不是改门控或继续V9/V10。

## 2. 实际执行与冻结条件

| Split | 原生几何 | Cases | Episodes | 拟合/评估样本 |
| --- | --- | --- | ---: | ---: |
| Train | circle | 60000–60063 | 64 | 3,151 |
| Validation | circle | 60064–60079 | 16 | 693 |
| Test | circle | 60080–60111 | 32 | 1,362 |
| OOD test | square | 61000–61031 | 32 | 1,475 |

144条轨迹、7,138个控制帧；127次到达、14次碰撞、3次超时，全部保留。收集策略是继承的合法观测ORCA teacher，不是成功轨迹筛选。固定间隔采样，不按预测误差挑帧。每个case只属于一个split，禁止随机拆帧泄漏同一轨迹。

输入是合法可见/保留轨迹的位置、速度、半径、age和测量有效性，以及当前机器人状态和当前其他合法行人。数值ID不进入探针。隐藏目标、上一控制步的 `_last_pref_vel`、未来真实位置仅作为标签。历史截止当前帧。

当前帧基线不是只看单个人：它也接收当前其他合法行人的状态。每个窗口臂使用相同233维输入布局、256个RBF landmark、同一ridge模型族，3,912个回归系数加截距；未使用的过去槽位置零。统计臂为18维历史摘要加当前特征，2,584个系数加截距。landmark行号相同，只在训练集标准化；16个超参数组合只用validation选择。

统计臂包含合法首次看见位置、窗口首个测量、速度均值/标准差、速度与位置变化、测量数和速度范围。首次看见位置是生命周期信息，**不限定在24帧内**。窗口臂最多24帧，即覆盖首尾间5.75秒。

目标方向从合法当前位置指向真实目标；真实目标距离小于0.25m时不计算方向角误差。为获得统一的1/2秒未来标签，预测探针排除距轨迹终止不足2秒的帧，因此不能代表全部临碰撞尾部。

## 3. 隐藏状态恢复与运动预测

下表为逐样本平均；2秒运动误差是endpoint displacement error，不是导航效果。

| 输入/规则 | circle目标角误差° | circle平滑状态误差m/s | circle未来2s误差m | square目标角误差° | square平滑状态误差m/s | square未来2s误差m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 10.16 | 0.1783 | 0.3156 | 25.74 | 0.2029 | 0.3964 |
| History3 | 9.54 | 0.1754 | 0.2974 | 26.42 | 0.2122 | 0.3749 |
| History6 | 9.98 | 0.1770 | 0.2868 | 26.68 | 0.2309 | 0.3851 |
| History12 | 9.51 | 0.1728 | 0.2710 | 27.06 | 0.2514 | 0.4031 |
| History24 | 8.51 | 0.1548 | 0.2615 | 35.45 | 0.3387 | 0.4694 |
| Simple history statistics | 4.02 | 0.1276 | 0.2655 | 47.44 | 0.4464 | 0.5848 |
| Current velocity/CV | — | 0.1774 | 0.3246 | — | 0.0570 | 0.2493 |
| Circle first-sighting goal prior + CV | 1.82 | 0.1774 | 0.3246 | 52.53 | 0.0570 | 0.2493 |

CV行以当前实测/保留速度估计平滑状态、线性外推未来；**不声称该速度就是ORCA内部状态**。first-sighting规则的目标估计为首次合法观测位置的相反数，不读取真实出生位置；它是circle生成器专用规则，square一列是其边界检查。

circle目标位置误差：Current 1.0903m、History24 0.6343m、统计摘要0.2583m、首次看见位置规则0.0936m。这表明大量可恢复目标信息很可能来自生成器规律；不能把这一现象解释成复杂交互记忆。

按case先平均再配对bootstrap，Current减History24的2秒误差为 **+0.0475m，95%区间[0.0365, 0.0581]**；square为 **−0.0824m，区间[−0.0970, −0.0671]**。这与逐样本表使用不同权重。区间为探索性结果，没有多重比较校正。

circle保留遮挡样本221个：2秒误差Current 0.4909m、History24 0.3928m、统计0.3853m、CV 0.5346m。square保留遮挡样本146个：对应0.7373/0.7065/0.8350/0.5876m。不能把circle的增量推广到所有遮挡场景。

square的1秒误差Current 0.1437m、History24 0.1866m、CV 0.0806m。**更复杂的探针不是天然更强的baseline。**

## 4. 动作后果检查：不是持续一个动作数秒

每个root枚举原生80个动作，**只执行一个0.25s控制步**，随后执行同一个合法ORCA continuation到原生终止。另一套continuation仅关闭机器人teacher的TTC brake，行人动力学保持不变。根部teacher平滑状态由合法过去观测重建。

实际参考是该continuation下的折扣原奖励 `Q^pi`，gamma=0.99，progress reward仍为0。没有用最大进度当正确动作，没有声称得到最优Q。root动作不加新的风险评分、过滤或平滑；该诊断不是整套学习导航器的部署成绩。

Current/History24/Statistics分别估计行人的隐藏目标与上一时刻平滑状态，用同一个原生动力学模型产生预测世界。预测世界选动作，**真实世界只评估该动作**。CV单独使用恒速行人世界。truth只作为离线参考。

预测世界采用单点隐状态估计，没有传播后验；这一估计器失败不等于所有history-aware planning都失败。每个root所有人可见，以避免偷偷补入当前隐藏位置；因此它隔离的是潜在运动状态，不直接检验遮挡期行动。

### 4.1 预先固定的主诊断

Cases：circle 60080–60083、square 61000–61003；每case首次满足时间≥6s、五人全部可见、距机器人目标>1m的帧。均有合法root，无替换。

| 方法 | 原continuation平均regret | 关闭brake平均regret |
| --- | ---: | ---: |
| CV | 0.00206 | 0.00000 |
| Current probe | 0.01700 | 0.01702 |
| History24 probe | 0.00958 | 0.00939 |
| Statistics probe | 0.02216 | 0.02118 |

regret为真实 `max_a Q^pi - Q^pi(a_selected)`。8个root的80个动作在两套真实continuation下全部成功；CV在16个root×continuation组合中15个达到最高回报。因此这批root几乎没有安全改进空间，**不能据此否定整个benchmark的时序headroom**。

### 4.2 补充的近身相遇诊断

主诊断发现覆盖较容易的状态后，新增一个明确标记的探索性cohort，原结果不删除、不替换。仅按当前合法几何选状态：时间≥2s、五人全可见、距目标>1m、最近人体表面距离≤0.8m。每个case取首次合格帧，每种几何取case编号最小的4个合格case。**没有按方法结果、未来真值或终止类型筛选。**

Roots：circle 60080@14、60082@36、60084@30、60085@8；square 61000@13、61001@23、61002@16、61003@15。`@`之后为控制帧序号。60082@36与主诊断重复，不能当独立证据再加权。

下表为近身root的平均regret，越小越好；每列仅4个root。

| 方案 | circle原continuation | circle关闭brake | square原continuation | square关闭brake |
| --- | ---: | ---: | ---: | ---: |
| CV | 0.01274 | 0.00879 | 0.02046 | 0.02042 |
| Current | 0.03898 | 0.03155 | 0.01649 | 0.01627 |
| History24 | 0.01487 | 0.01901 | 0.01838 | 0.02002 |
| Statistics | 0.02795 | 0.02614 | 0.01266 | 0.01436 |
| Current + true goal only | 0.00000 | 0.00000 | 0.00183 | 0.00000 |
| Current + true preferred state only | 0.02964 | 0.03155 | 0.01838 | 0.01846 |
| Legal first-sighting goal prior | 0.00000 | 0.00000 | 0.01623 | 0.01981 |

所有方案选出的动作都最终成功，没有已验证的SR/CR优势。square 61000中，80个参考动作有11个碰撞，其余近身root的80个参考动作全部成功。

History24相对Current：原continuation 1胜/2负/5平，关闭brake 2胜/2负/4平。相对CV：2胜/2负/4平、3胜/3负/2平。**没有稳定超越简单规则。**

还原真实目标后，circle平均剩余到达时间由8.4375→7.1875s、8.3125→7.3125s；square由6.625→6.125s、6.6875→6.125s。还原平滑状态没有稳定对应收益。这隔离出目标估计的局部效率价值，但该oracle不可部署。

合法首次看见位置规则在circle的8个root×continuation组合中全部达到参考最高回报；相对CV的平均剩余时间分别减少0.4375s和0.3125s。**同回报不等于同动作、同安全距离或同轨迹。** 该规则在square不能恢复这种优势。

## 5. 事实、推断、未知

**已证实事实：** 隐藏目标和平滑状态确实进入行人动力学；circle合法历史改善这些指定探针的恢复/预测；目标oracle改变部分动作后果；两数的circle规则即可实现局部oracle回报；长历史的square泛化失败；平滑状态oracle未提供稳定收益。

**有证据支持的推断：** 目前较明确的信息主要是目标/路线恢复，而不是“改向后有选择地擦除旧证据”；circle长历史收益有较强生成器先验成分。共享后续控制有能力修正根部动作，但没有独立实验确认这是控制频率造成的。

**尚未证明：** 任意最强Current模型仍无法恢复目标；历史在一般square场景能稳定补上该缺口；隐藏目标恢复能提升原学习导航器SR；KDA比GRU或简单目标滤波更适合；ORCA内部状态在所有场景没有价值。当前数据不支持这些更强结论。

## 6. 下一步与停止条件

本轮实测结束，不自动训练新导航架构。推荐下一步仅做**跨生成器的目标/运动隐状态恢复**：使拟合数据包含原生square，采用相对运动表征，并保留完全未用作选择的case；与CV、短窗统计和简单目标滤波比较，再接同一root-step continuation诊断。新的拟合/验证边界必须重新固定。

若历史仅学会对径目标规则，或所有动作收益均被合法简单统计吸收，不再用该结果支撑复杂记忆。若合法历史在非规则目标场景同时改善预测和原奖励后果，才有依据开展统一IL→RL的GRU/KDA竞争；无需先证明GRU必然失败。

## 7. 验证与复现

本轮全程本地4个CPU worker，没有下载数据、没有占用4090服务器磁盘。新增诊断没有修改冻结的科学源码；其SHA256仍为：

fb1cfdc5abda8b0af9df66aeae4bd4d1674894ea6b43eb5c55de3265f6889bd5

100项测试，97通过、3项原有可选legacy资产测试跳过。收集器去掉未使用的随机导航网络后，case60080的全部动作、奖励和合法观测与原始采集逐项完全一致。所有root单步恢复的最大状态差≤4.77e-7，奖励差为0。

主诊断6,400个分支；近身五臂6,400个分支；两个单隐变量oracle追加2,560个；合法目标规则追加1,280个。**合计16,640个候选分支，不是16,640个独立benchmark episodes**；动作证据来自10个case的15个不同root、两种continuation。原始逐动作return/terminal/distance/time均保留，不能按分支数夸大统计样本量。

数据和拟合参数在本地outputs/latent_information，不上传大文件。主要原始SHA256：

| Artifact | SHA256 |
| --- | --- |
| episodes.pt | c23bac187c537527a067ad0863ac7f5d9040ea5e90534941f040a31fadc2904e |
| probes.pt | e5718119f99c7c35a3f3970f37d1d2cd3a0d5b97163859e61f3baab8575164b8 |
| interaction_birth_decision_branches.pt | 045f84746aa2bd7b85d2feb4d83e8147679db80b8930997fb78a42cbf12b0a68 |

探针依赖是可选的audit extra，不进入导航runtime。本地执行环境为NumPy1.23.5、SciPy1.3.3、scikit-learn1.0.2、PyTorch2.1.0，使用/usr/bin/python；未在别的版本组合上声称复现。以下顺序从固定protocol复现；保留原始结果时应使用不同的output目录，避免覆盖。

```bash
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.latent_information collect --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_information probe
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --components --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --birth-prior --workers 4
```
