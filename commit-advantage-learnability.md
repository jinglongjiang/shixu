# Commit-vs-Replan：可学习性与四臂闭环结果

更新：2026-10-05。已完成384条五人标签轨迹、1,259个配对root、3,072条fresh闭环。

**冻结裁决：LEARNED_INITIATION_NOT_CONFIRMED。不是完全无收益，也不是METHOD_ENTRY_FOUND。**

## 1. 结论先说清

- 线性gate通过预先冻结的离线开发准入；MLP没有通过跨checkpoint留出。这个准入不是方法成功。
- fresh完整闭环中，MLP相对Parent到达677→688、碰撞23→20，原折扣回报均值0.434260→0.441807。确有受控的正向结果。
- square低进度timeout54→48、到达312→317，但破坏8个原成功；冻结上限是7个。因此MLP只通过8/9项Parent对照门槛，不能因只差一个就修改标准。
- 20人square：Parent到达91，MLP90；该格原回报也下降。尚未建立所要求的密集场景泛化优势。
- 线性gate也通过8/9项，但破坏19个square原成功，而且20人square明显恶化。不能用整体SR上升掩盖这一点。
- 固定V0.1在本块square没有复现上块的净改善。87000块的正证据保留，不能把它扩大成稳定跨case结论，也不能用本块去改写旧数字。

本轮没有调阈值、换checkpoint、改duration、重训critic或启动新的gate版本。结果封账，停止本次自动研发。

## 2. 这次真正实现了什么

Parent还是原CV后继模型、actor价值网络、80个原生动作、原reward/filter/risk/history和平滑合同。四个原权重419/443/467/491均为原五人circle IL50/在线MC3000资产，本轮没有重新执行IL或RL。

新增部分只决定是否启动最多2秒的持有：

> 当前Parent选动作 → 估计一次commit相对Parent续跑的回报差 → 大于0时允许启动 → V0.1安全释放 → 恢复Parent。

不是新导航动作策略，不是新critic，不学习duration或release。线性模型19参数；MLP是146→16→1，共2,369参数。四个Parent共用同一个case-only线性模型和同一个case-only MLP，**不是四个独立训练的gate**。

沿用每步一次0.3旧命令+0.7固定grid的平滑；每步0.25秒更新合法观测、tracker和真实history。持有动作被原margin检查挡住、或80动作全部没有clearance≥0.2m时，同一步release并执行最新Parent动作。原Parent的all-unsafe fallback不变。

启动后最多8步；资格条件清除再出现才可再次启动。不在安全持有期间重新学习release或重算gate。部署不使用未来、隐藏人目标、内部ORCA状态、case/seed/geometry编号。

## 3. 标签合同与数据

只在原生五人circle采集。square及10/20人没有参与拟合或模型选择，保持原训练域并测试OOD。这是本次设定，不是声称五人训练必须只用circle。

96个新case90000–90095 × 四个冻结Parent，384条完整轨迹。选root不看方法结果或Parent终局：固定2/6/10秒，加首次“过去2秒进度≤0.2m”的合法时刻；去重、距自身目标>1m。

每个root恢复相同float64世界、合法history、tracker和NumPy随机状态：

- A：原Parent一直续跑到终局。
- B：当前Parent grid最多持有8步，沿用V0.1释放，之后**永久Parent**。
- 标签y=G(B)−G(A)，使用原gamma=0.99及原环境奖励，从root开始折扣。

这是指定Parent continuation下的一次控制干预差值，**不是最优advantage，也不是反复启动gate的Q函数**。模拟器隐状态仅用于恢复世界与生成离线标签，不进入gate输入。

384条Parent之外另有2,518条恢复root的续跑分支，不能将它们计成独立场景样本。Parent/V0.1没有使用新增监督标签；线性与MLP获得相同标签。本轮单列了额外数据/仿真成本，尚未做等额新增交互预算重训Parent的论文级比较。

三个89999预检root不进入数据集。全部1,259个正式root的A分支与原Parent剩余轨迹逐动作一致，终局/折扣回报一致；384个输入归档哈希回查通过。

| 划分 | case | Parent轨迹数 | root数 | y>0.005 | y<−0.005 | 其余 |
|---|---|---:|---:|---:|---:|---:|
| 训练 | 90000–90063 | 256 | 840 | 197 | 449 | 194 |
| 验证 | 90064–90079 | 64 | 207 | 57 | 105 | 45 |
| 测试 | 90080–90095 | 64 | 212 | 56 | 123 | 33 |

同case的全部root不跨划分。每个(seed,case)总权重相同，防止root多的轨迹主导损失。四次leave-checkpoint-out同时保留未见seed和未见case；另做同checkpoint、未见case的case-only检验。测试标签不用于refit。

训练840个root中，commit只产生3个timeout→success，却有16个success→timeout；验证分别1个和2个。标签域没有碰撞分支，不能声称训练时学到了全面安全收益判别。训练标签以正常成功轨迹为主，是重要限制。

## 4. 输入及训练没有变复杂

18个合法标量：自身目标距离、过去2秒进度、窗口完整性、当前grid/平滑命令、所选score、top1-top2 gap、预测clearance、安全候选比例、近期grid变化、机器人速度/命令变化、可见比例、track年龄、当前间距、目标方向一致性。

MLP另读取Parent已计算的128维“所选候选在原value head之前”的特征。它是合法历史及CV假想后继的模型表征，**不是额外真值、不是单纯current embedding**。线性模型只用18个标量。

线性ridge固定alpha=1；MLP固定16宽度、Adam0.001、200次full-batch更新、按验证MSE保存。仅训练集标准化，未搜索特征/超参。case-only MLP选中的epoch为42。

不同Parent的潜特征坐标未证明对齐。MLP跨checkpoint失败不能唯一归因于“advantage不可学习”；语义线性模型是避免这种误判的独立对照。

## 5. 离线准入：弱信号必须按弱信号报告

表中收益是episode平衡的“一次被gate选中干预的真实回报差”均值，不是线上SR。

| 模型 | 未见case收益 | 同时未见case/seed收益 | 跨seed收益非负 | 跨seed平衡准确率 | 离线准入 |
|---|---:|---:|---:|---:|---|
| 语义线性 | +0.003781 | +0.001206 | 3/4 | 51.69% | 通过冻结开发门槛 |
| 小MLP | +0.003163 | −0.000536 | 2/4 | 48.61% | 未通过 |

跨seed线性收益419/443/467/491：+0.005278/+0.001162/−0.001931/+0.000316。MLP：−0.000133/+0.000553/−0.002564/0。

线性模型的未见case收益去掉90093后变成−0.000091，显示正收益集中。这个补充检查不是事后改Gate，而是说明信号强度；不能宣称已可靠识别收益。原冻结准入是任一模型case-only及联合留出收益>0.001、至少3/4 seed非负、正负标签各覆盖至少4个测试case、额外碰撞≤1pp。线性通过，因此继续完整闭环，不仅挑它跑，也保留MLP作比较。

## 6. Fresh完整闭环

最初拟用91000块，冻结前查到旧PaS使用记录；没有运行新方法就弃用。正式固定99000–99031，本地既有JSON case/start字段未见使用。

Parent/V0.1/线性/MLP × 四个Parent权重 × 5/10/20人 × circle/square × 每格32case，共3,072条、768组四臂相同初始世界。没有结果导向停跑，全部跑到原生终局。99000块现已消耗，不能再称fresh。

### 总体结果：每臂768条

| 指标 | Parent | 固定V0.1 | 语义线性gate | MLP gate |
|---|---:|---:|---:|---:|
| 到达 | 677 | 671 | 684 | 688 |
| SR | 88.15% | 87.37% | 89.06% | 89.58% |
| 碰撞 | 23 | 22 | 22 | 20 |
| 超时 | 68 | 75 | 62 | 60 |
| 原折扣回报均值 | 0.434260 | 0.418251 | 0.440631 | 0.441807 |
| 成功episode平均时间 | 16.763s | 17.493s | 16.648s | 16.863s |
| 全episode最小间距的均值 | 0.363225m | 0.365928m | 0.333193m | 0.345552m |
| 全块最差间距 | −0.090052m | −0.090052m | −0.138470m | −0.090390m |

成功用时的样本构成不同，不直接当效率优势。MLP与Parent共同成功的359个circle配对平均−0.019s；304个square配对平均+0.227s。

### 六格终局：每格每臂128条，S/C/T为成功/碰撞/超时

| 场景 | Parent S/C/T | V0.1 S/C/T | 线性 S/C/T | MLP S/C/T |
|---|---|---|---|---|
| 5 circle | 123/0/5 | 123/0/5 | 126/0/2 | 124/0/4 |
| 10 circle | 125/1/2 | 121/1/6 | 124/0/4 | 127/0/1 |
| 20 circle | 117/6/5 | 116/5/7 | 121/3/4 | 120/3/5 |
| 5 square | 109/0/19 | 110/0/18 | 117/0/11 | 112/0/16 |
| 10 square | 112/1/15 | 114/1/13 | 113/3/12 | 115/1/12 |
| 20 square | 91/15/22 | 87/15/26 | 83/16/29 | 90/16/22 |

MLP在5/10人square各净增3个成功，20人净少1个。20人square原Q：Parent0.265783、线性0.223533、MLP0.249994。**不能写成5→10→20泛化全部成立。**

### Square主目标及circle保护

| 指标 | Parent | V0.1 | 线性 | MLP |
|---|---:|---:|---:|---:|
| square成功 | 312 | 311 | 313 | 317 |
| square低进度timeout | 54 | 55 | 50 | 48 |
| square碰撞 | 16 | 16 | 19 | 17 |
| square Q均值 | 0.397277 | 0.381582 | 0.393710 | 0.400264 |
| circle成功 | 365 | 360 | 371 | 371 |
| circle碰撞 | 7 | 6 | 3 | 3 |
| circle Q均值 | 0.471244 | 0.454920 | 0.487553 | 0.483351 |
| square失败→成功 | — | 11 | 20 | 13 |
| square成功→失败 | — | 12 | 19 | 8 |
| circle失败→成功 | — | 4 | 12 | 12 |
| circle成功→失败 | — | 9 | 6 | 6 |

MLP square的8个损害为7个成功→timeout、1个成功→collision；circle6个为4个timeout、2个collision。MLP square13个挽救均满足原0.02m全episode间距筛选，来自11个独立case编号。circle12个挽救中11个满足该筛选；不将不满足者称作合格安全挽救。

MLP只比线性多4个总体成功，总体Q多0.001176；尚未证明非线性/潜特征具有稳定额外收益。线性的circle Q反而更高。

## 7. 原九项门槛，不改口径

| 原冻结标准 | 线性 | MLP |
|---|---|---|
| square低进度timeout净减少 | 减少4，通过 | 减少6，通过 |
| ≥4独立case合格挽救 | 15个case，通过 | 11个case，通过 |
| ≥2人数配置SR与主目标同时净改善 | 5/10人，通过 | 5/10人，通过 |
| ≥3/4 Parent seed主方向非负 | 3/4，通过 | 3/4，通过 |
| square碰撞增加≤1pp | +3/384，通过 | +1/384，通过 |
| square原成功破坏≤2pp，即≤7个 | 19/384，失败 | 8/384，失败 |
| circle原成功破坏≤2pp，即≤7个 | 6/384，通过 | 6/384，通过 |
| circle SR净损失≤1pp | 净增6，通过 | 净增6，通过 |
| circle碰撞增加≤1pp | 净少4，通过 | 净少4，通过 |

MLP还在square SR/Q上超过固定V0.1与线性；但这一增量条件不能替代失败的原保护标准。两个学习臂均8/9，不升级METHOD_ENTRY。

| seed | 线性square成功净增 / 低timeout净减 | MLP对应指标 |
|---|---:|---:|
| 419 | +2 / +4 | +3 / +3 |
| 443 | −4 / −3 | −1 / +1 |
| 467 | +1 / 0 | +1 / 0 |
| 491 | +2 / +3 | +2 / +2 |

四行是四个冻结Parent训练seed的评估，不是四个gate训练seed实验。不能拿3/4的方向计数充当统计显著性。

## 8. 正确性与成本

224 tests通过，3个旧legacy路径相关测试跳过。3072条轨迹逐条校验原折扣奖励、0.25秒时钟、实际grid的单次平滑、同一步release及保存输入到gate输出的数值一致性。

实际控制步239,772。没有持有blocked/all-unsafe动作。未发生持有的Parent配对整条轨迹相同：V0.1 265组、线性11组、MLP42组。大多数episode都会启动至少一次学习gate，不应把它宣传为只针对少数失败的稀疏接管。

标签采集墙钟201.1秒；3072条闭环493.3秒。本地RTX3060、四进程、CPU线程1，未向4090下载资产。

| 成本 | 线性 | MLP |
|---|---:|---:|
| 新参数 | 19 | 2369 |
| gate文件 | 1924 B | 15424 B |
| 仅gate CPU调用均值，1000次 | 0.0145ms | 0.0347ms |
| 原样确定性拟合复算计时 | 0.0104s | 0.5355s |
| fresh并发完整predict均值 | 5.796ms | 6.252ms |

原十次fit未逐模型计时；表中拟合时间是只用原train/val的确定性权重复算，参数与原文件逐项完全相同，**没有重选checkpoint、没有替换结果**。不能把它冒称原作业精确计时。

同一冻结root/state/history下，100次轮换CUDA同步计时的score路径：Parent2.795ms、线性score+特征+gate3.441ms、MLP3.468ms。增加约0.65–0.67ms，主要路径不是仅0.035ms的MLP调用。该测量排除额外wrapper预检查/历史写入，不是完整机器人部署延迟。

完整predict均值还受并发与轨迹构成影响：Parent5.429ms、V0.1 5.694ms。因此现在只能报告代价，不声称成本已被收益充分证明值得。

## 9. 已证实、推断、未证明

| 等级 | 判断 |
|---|---|
| 已证实 | 固定五人监督训练的两个gate在线改变了commit启动；有实际终局和原奖励差分，不只是动作敏感性 |
| 已证实 | MLP在本fresh块总体、square主指标上有正增量，同时保留8个square成功损害和20人square退化 |
| 已证实 | 线性是很强的简单替代；MLP没有取得通过冻结保护标准的完整优势 |
| 已证实 | MLP未通过离线seed留出，却在线有正结果；弱离线probe不能当整条机制的终局判决 |
| 有证据支持的推断 | 使用commit相对replan的收益监督，比固定低进度trigger更值得保留为研究机制 |
| 尚未证明 | 原策略失败唯一由高频replanning造成；剩余损害唯一来自启动，而非时长/动作/重复启动交互 |
| 尚未证明 | 一次Parent-tail标签与重复部署的差分就是当前损害的原因；它们合同确有差别，但尚无因果消融 |
| 尚未证明 | 非线性gate/潜特征比19参数语义线性具有可重复、跨gate训练seed的优势 |
| 尚未证明 | 20人square泛化、完整安全保护和成本收益足以支撑SCI主张 |

## 10. 新颖性边界与唯一下一步

[AFST原论文](https://arxiv.org/abs/2108.06161)已将自适应执行时间纳入机器人导航SMDP。一般options也已研究启动/终止；本轮没有完成exact-prior审计，不能把“按advantage启动持有”直接叫新算法，更不能声称首次自适应动作持续时间。

**建议的唯一下一步是：不改这两个gate、duration或阈值，先做一块独立fresh四臂确认，沿用全部保护门槛并保留20人square这一已暴露薄弱格。** 本轮不自动执行。

理由：现在有真实闭环增量，但MLP保护失败、线性替代强、跨checkpoint离线信号弱。独立确认能区分“值得继续的方法信号”与“case块上的开发正结果”；继续加结构或调trigger反而会破坏可解释性。

若确认仍有square原成功损害超标或密集square退化，就不能宣称合格方法；届时应明确诊断一个具体机制再决定一次有依据的修改，不能连续试阈值直到赢。若确认合格，再做gate训练seed、潜特征/语义输入和强简单方法消融，并精确核查先例。现在不提前实施这些分支。

## 11. 归档与源码

父体提交e2f0621；标签/学习阶段源码及冻结数据提交ed6be93，独立分支research/commit-advantage-learnability。label阶段源码哈希保留；新增在线module不改旧source字节，在线协议通过排除新module回查原label source，再单独冻结完整新source。

保留384个本地输入snapshot、各root两分支续跑命令、3072个完整trace及episode元数据。Git只上传源码、一个本报告、gate权重、标签特征/回报及精简冻结结果，不上传全部snapshot/trace。隐藏真值恢复文件不属于部署输入。

主证据：

/home/abc/workspace/shixu/outputs/commit-advantage-probe/protocol.json

/home/abc/workspace/shixu/outputs/commit-advantage-probe/dataset.json

/home/abc/workspace/shixu/outputs/commit-advantage-probe/learnability.json

/home/abc/workspace/shixu/outputs/online-commit-advantage/protocol.json

/home/abc/workspace/shixu/outputs/online-commit-advantage/summary.json

/home/abc/workspace/shixu/outputs/online-commit-advantage/validation.json

/home/abc/workspace/shixu/outputs/online-commit-advantage/runtime-profile.json

本地只读复核：PYTHONPATH=vendor:. /usr/bin/python -m experiments.online_commit_advantage validate。重新采集标签应使用ed6be93的干净源码与原critic资产；不能在已保存probe上覆盖训练，也不能把已使用case重新称作独立测试。
