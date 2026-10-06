# KDA and Temporal Experiments

Historical KDA versions and matched memory comparisons. No stable KDA navigation advantage was established; this archive does not authorize another KDA version.

## Archive Policy

This is a classification-only consolidation. Original stage text, numbers, negative results and withdrawn claims are preserved byte-for-byte below. Later closure reports supersede earlier proposed next steps. The historical README is retained in thematic sections. Snapshot variants are labeled by their original paths. Frozen protocols and autoresearch_nav/program.md are not changed.

## Stage Index

- [OCCLUSION_KDA_INTERIM_REPORT_20261004.md](#stage-1)
- [README.md (historical README section 1)](#stage-2)
- [outputs/forecast_control_remote_backup_20261004/README.md (historical README section 1)](#stage-3)


---

<a id="stage-1"></a>

## Source: OCCLUSION_KDA_INTERIM_REPORT_20261004.md

Original full-source SHA-256: dbe1de6a13ae2f3aec1132287d3752cbe760bd6d581b6dc86ed00546cb5d4313

<!-- BEGIN PRESERVED SOURCE -->
# KDA 遮挡导航研发中间报告

统计截至：2026-10-04 10:16，韩国时间。后续运行状态会变化，本文先保留这一时点的完整快照。

## 1. 先说结论

**还没有得到你要求的结果：在 5 人环境完成训练后，KDA 在 5/10/20 人测试中稳定赢过健康基线，并且不能只靠更保守、更慢来换取少碰撞。**

目前拿到的是三个阶段性成果，不是新算法成功：

1. 从本地 camrl 整理出了可训练、可保存权重、可重载测试的干净时序导航母体，并上传到 shixu。
2. 完成了多轮真正的 IL + 在线 MC 价值学习和闭环测试，不是用未训练网络或 forward 成功代替结果。
3. 排除了几个看似合理、但没有被实验支持的解释；目前仍不知道怎样把 KDA 的记忆能力转化为稳定的导航优势。

**需要警惕的偏离：版本已经试得不少，但“遮挡任务中究竟缺少什么决策信息”仍没有充分做透。继续堆版本，不等于研究更深入。**

## 2. 时间线：昨天到现在做了什么

以下时间来自本地 Git 提交记录。它们表示代码或阶段结论的记录时间，不等于每个训练任务的精确起止时间。

| 时间 | 工作 | 得到什么 |
| --- | --- | --- |
| 10 月 3 日 13:33 | 提纯本地 camrl，保留原价值 lookahead 和 IL/MC 路线 | 干净母体、原模型数值及动作一致性检查 |
| 15:05–15:44 | scene-first 与 actor-first 时序顺序比较，随后做一次共同输入接口修订 | actor continuity 接口可用，但没有稳定性能胜出 |
| 19:10–20:22 | 实现 GRU/KDA/GDN2 的逐人记忆，完成八臂、两种子训练 | generic KDA gate 有初步信号，显式运动证据版本更差 |
| 20:59–10 月 4 日 00:16 | 拆 gate，再训练静态缩放与动态 gate 四种子对照 | 早期 gate 优势没有稳定复现 |
| 03:13 | 正式建立真实人体遮挡的合法观测/保留轨迹接口 | 5 人训练、5/10/20 人重载权重测试开始 |
| 03:51–06:46 | 观测时钟、上下文写入、局部检索地址、候选私有更新等替换 | 多个局部机制版本完成，但没有合格优势 |
| 06:48–08:57 | 遮挡真值 shadow、物理运动记忆、时间间隔、到期轨迹和动作敏感性检查 | 记忆确实影响动作；不是已经证明动作更好 |
| 08:46–09:43 | 候选动作条件读取 V7，完整四种子比较 | read-only 比某个较弱 KDA 更新版本好，但仍输强基线 |
| 09:31 起 | 冻结 V8：四臂统一增加到 3,000 个在线回合 | 正在运行；只检验预算，不新增模型结构 |
| 10:12 起 | 检查真实观测训练与 CV 后继查询的接口差异，补做冻结消费者真值替换 | 暂不支持“大查询失配”解释；KDA 真值替换未提高 SR |

## 3. 现在实际使用的框架

主链保持为：

```text
实际可见观测
  -> 已见行人的身份对齐历史
  -> 逐行人共享时序模型
  -> 当前人机几何融合与人群聚合
  -> 一个标量价值
  -> 80 个原候选动作的一步后继评估
  -> 执行动作

ORCA 成功轨迹
  -> Monte Carlo 回报预训练
  -> 在线收集轨迹、回报回归更新
  -> 保存最终权重
  -> 重新加载权重做闭环测试
```

这里的“RL”准确地说是继承母体的在线 MC 价值精炼，不是 PPO、SAC，也不是另外造了一套训练范式。IL 学习的是专家轨迹的回报，不是动作分类式行为克隆。

一直没有加入双记忆、世界模型、VLM、部署教师或辅助预测网络。当前物理记忆版中没有额外的外部 memory gate；KDA 内部原生投影和门控属于其已有算子。

**这里的逐人记忆仍是有限窗口记忆。** 每次评分从最近 24 帧窗口重建每个行人的状态，真实输入前缀最多 23 帧，再进行当前/后继查询；没有把矩阵状态跨整个 episode 无限保存。约 6 秒窗口之外的观测不能直接残留在当前矩阵里。因此，不能把后期停滞直接解释成“很久以前的运动一直没有被忘掉”。早期动作仍可能影响后来访问的状态，但那是闭环轨迹效应，不是无限期记忆残留。

### 遮挡不是随机删人

- 使用 360 度人体中心射线遮挡，没有随机 dropout 或额外距离截断。
- 从未实际见过的人不能进入模型。
- 已经见过但暂时被遮挡的人，用最后测到的速度做 CV 保留，最多 2 秒。
- 实际测量可以写记忆；CV 保留和候选假想后继不写入真实观测记忆。
- 关联键只用来对齐轨迹，不作为数字特征送进网络。
- 原五人输入截断已经移除，各臂都可以处理 10/20 人；不是只给新模型增加人数。
- ORCA 专家和学生使用同一合法观测/CV 保留接口，没有把专家可见的未观测真值送入学生。

**重要环境边界：当前机器人 visible=false，行人之间使用 ORCA，但行人不因机器人动作主动避让。** 因此这轮可以研究遮挡导航，不足以验证强响应式人机互动或“历史让行关系”的完整主张。

## 4. 训练和测试到底怎么做

### 正式遮挡开发轮 V1–V7

| 项目 | 固定设置 |
| --- | --- |
| IL 数据 | 同一组 128 条合法观测 ORCA 成功轨迹，148 次尝试得到；不是模型测试后挑出的成功案例 |
| IL | 50 epochs，batch 256，学习率 1e-4 |
| 在线 MC | 每模型 1,000 回合，每回合 4 次更新，batch 256，学习率 1e-5 |
| 训练环境 | 仅 5 人 circle |
| 时序输入 | T=24，width=128，2 层，零填充，可变人数 |
| 动作/奖励 | 同一 80 个移动动作、gamma=0.99、原奖励；无新进度奖励 |
| 开发训练种子 | 419、443、467、491 |
| 测试 | 5/10/20 人 × circle/square × 每格 16 cases，每模型 96 episodes |
| 模型选择 | 固定预算结束的最终权重；不挑最高 SR 的中途权重 |
| 新确认资源 | 521/547/569/593 和 cases 40000–40031，尚未使用 |

主指标是等权的 10/20 人 SR。5 人 SR、碰撞、超时、成功用时、路径、最小间距和推理成本同时报告。

研发通过规则预先固定：相对各强对照，主 SR 至少 +3 pp，至少 3/4 配对种子同方向，且 CR 不能增加超过 1 pp、timeout 不能增加超过 2 pp、成功用时不能明显恶化。这里不是要求超过 90%；是防止把碰撞转成超时或偶然单种子优势叫作成功。

成功用时只来自成功的 episodes，存在不同方法成功集合不同的问题，不能替代严格的配对进度结论。

### 已完成多少，哪些不能重复计数

V1–V7 共完成 **88 个独立训练臂×种子运行、8,448 个最终权重测试 episodes**。比较表中复用的旧基线没有当成新训练或新测试重复计数。

之前非正式遮挡阶段的 memory pilot 和 scale follow-up 另有 32 个运行、3,072 个测试 episodes。它们与正式遮挡结果分开，不能混在一起计算最终平均值。早期 IL 快照有合法复用，日志中的 50 个 IL epoch 不代表每轮都重新消耗了同样的 IL 训练时间。

这些数量说明实际进行了训练和测试，不说明科学证据一定足够强。多个局部变体反复使用同一开发集，必须依靠未见种子和未见 cases 做最终确认。

## 5. 你之前问的 79.17/79.43/79.69 是哪里来的

这组是指定遮挡课题之前的四种子缩放对照，不是正式遮挡 V1–V7 的结果。

| 模型 | 总 SR | CR | Timeout | 成功用时 |
| --- | ---: | ---: | ---: | ---: |
| Actor-GRU | 79.17% | 10.16% | 10.68% | 16.62 s |
| KDA 不缩放 | 76.82% | 11.72% | 11.46% | 22.21 s |
| KDA 静态缩放 | 79.43% | 9.64% | 10.94% | 20.96 s |
| KDA 动态 gate | 79.69% | 10.94% | 9.38% | 19.43 s |

每个模型都完成 50 个 IL epochs 和 1,000 个在线 MC episodes，再加载最终权重测试。四种子为 307/331/359/383。

动态减静态的四个 SR 差值是 -1.04/0.00/-7.29/+9.38 pp。平均 +0.26 pp，配对 90% 区间 [-7.83,+8.35] pp：**没有稳定胜出，也不能据此宣称等效。**

更早两种子得到的 generic gate 84.90% 没有在这轮稳定复现。运动 evidence 去掉后 0/76 个诊断动作改变，不能解释成“学会了识别改向并主动忘记旧记忆”。这也是后来停止围绕 gate 扫变体的原因。

## 6. 正式遮挡 V1–V7：具体改了什么、结果如何

### 6.1 每轮只验证一个主要结构问题

| 版本 | 主要替换 | 试图回答的问题 | 结果解释 |
| --- | --- | --- | --- |
| V1 | 测量写入、保留轨迹读取、逐人 KDA | 原时序接口接入真实遮挡后能否胜出 | KDA 没赢 current-CV；碰撞更多 |
| V2 | 读取按观测时钟共享，随后融合候选几何 | 候选假想步是否不该控制记忆时间推进 | KDA 超时明显增加，具体版本不成立 |
| V3 | 将原有 attention 移到时序前，仍逐人保存输出 | 遮挡期间是否需要保存之前看见的邻居上下文 | GRU 改善，KDA 反而退化；不是证明上下文无用 |
| V4 | 保留 actor 自身残差，局部特征生成 q/k，context 生成内容 | 上下文是否使 KDA 检索地址过度相似 | 地址相关性可改变，但没有导航优势；所有臂重新训练 |
| V5 | 在每个候选的私有副本做一步更新，不写回真实历史 | read-only 是否缺少最后一步转换 | measured-only 和 CV-pseudowrite 都未赢强 GRU |
| V6 | 物理行人历史先入记忆，当前机器人目标/几何后融合 | 将历史存储与候选决策分开，间隔语义是否有效 | 基本功能正常；elapsed clock 没有稳定收益 |
| V7 | 当前人机候选特征先形成 query，再读物理记忆 | 候选相关读取能否优于同一历史摘要 | read-only 赢较弱 KDA branch，但仍输强对照 |

V4 改了共同当前特征接口，因此它的 current/GRU 也重训，不能把这一轮较弱的 GRU 当作全局最佳基线。

### 6.2 主要数字

下表都来自已完成 comparison.json。不同版本不是新独立测试集；跨版本数字只用于研发诊断，不作确认性统计。

| 版本/模型 | 总 SR | CR | Timeout | 主 10/20 人 SR |
| --- | ---: | ---: | ---: | ---: |
| V1 current-CV，无时序 | 81.77% | 4.95% | 13.28% | 79.30% |
| V1 KDA | 78.39% | 11.20% | 10.42% | 75.78% |
| V2 观测时钟 KDA | 63.54% | 5.99% | 30.47% | 64.45% |
| V3 上下文 GRU | 85.94% | 5.99% | 8.07% | 85.55% |
| V3 上下文 KDA | 62.76% | 12.24% | 25.00% | 60.16% |
| V4 同接口 GRU | 77.08% | 5.47% | 17.45% | 74.22% |
| V4 普通 KDA | 72.40% | 10.94% | 16.67% | 67.58% |
| V4 局部地址 KDA | 68.75% | 11.72% | 19.53% | 62.50% |
| V5 私有更新 KDA | 75.00% | 10.16% | 14.84% | 71.48% |
| V5 CV-pseudowrite KDA | 77.34% | 10.68% | 11.98% | 74.61% |
| V6 物理记忆 GRU | 87.50% | 4.43% | 8.07% | 83.59% |
| V6 物理 KDA，单位时钟 | 80.99% | 3.65% | 15.36% | 76.56% |
| V6 物理 KDA，elapsed clock | 80.99% | 2.86% | 16.15% | 76.95% |
| V6 物理 KDA，CV-pseudowrite | 83.07% | 5.47% | 11.46% | 78.12% |
| V6 同容量 KDA，零真实历史 | 86.72% | 6.51% | 6.77% | 83.59% |
| V7 同位置候选 query GRU | 87.24% | 5.47% | 7.29% | 82.81% |
| V7 KDA 私有 query 更新 | 71.61% | 8.07% | 20.31% | 70.31% |
| V7 KDA read-only query | 75.78% | 9.90% | 14.32% | 74.61% |
| V7 read-only，零真实历史 | 81.25% | 5.99% | 12.76% | 77.34% |

“零真实历史”不是完全无信息：仍有当前合法 CV 轨迹、age、间隔和相同容量的时序算子。它检验真实历史的增量，而不是裸传感器输入。

V6 elapsed 相对单位时钟的四种子主 SR 差值：-9.38/+4.69/+7.81/-1.56 pp，平均 +0.39 pp。

V7 read-only 相对同位置 GRU：-26.56/-4.69/0.00/-1.56 pp，平均 -8.20 pp。没有哪个版本满足既定通过规则。

## 7. 已做的诊断：哪些解释站得住，哪些站不住

### 7.1 真实遮挡和合法重现确实存在

128 条专家轨迹有 6,245 个控制帧、31,225 个 person-frames；其中 5,438 个被遮挡、4,205 个是可保留的已见隐藏人，903 次重新出现。

但是，隐藏 CV 位置误差均值约 0.0765 m，90 分位约 0.2303 m；距机器人 2 m 内的平均误差约 0.0272 m。**这套原生环境确实有遮挡，但近距离 CV 已经相当强。**

“遮挡普遍存在”与“这个 testbed 给我们的记忆机制留下很大提升空间”是两个问题，目前后者证据不足。

### 7.2 记忆确实影响动作，不只是给所有候选加常数

四种子共 626 个均匀冻结状态，338 个有保留隐藏行人：

- 只去掉隐藏行人的记忆，79/338 个动作改变，23.37%。
- 去掉全部真实历史，344/626 个动作改变，54.95%。

这否定了“记忆完全没被消费”的强解释，但不能说明改变后更正确。

### 7.3 在同一已训练 KDA 中，移除记忆会损失闭环表现

每格预先固定前两个 cases，每种子 12 episodes，共 48：

| 冻结模型干预 | SR | CR | Timeout |
| --- | ---: | ---: | ---: |
| 原始记忆 | 83.33% | 6.25% | 10.42% |
| 只屏蔽隐藏人的记忆 | 79.17% | 2.08% | 18.75% |
| 屏蔽全部记忆 | 54.17% | 6.25% | 39.58% |

屏蔽全部记忆四种子都下降；屏蔽隐藏记忆两降、一升、一平。它证明该模型依赖历史，但干预可能是分布外输入，**不能推翻单独训练的零历史模型更强这一结果。**

### 7.4 补准被遮挡的当前状态，没有出现大的冻结消费者收益

最新诊断保持 checkpoint、cases、奖励、动作和其他消费者不变，仅用离线真值替换已见、仍保留的隐藏人当前位置/速度。未见人不加入，tracker 不写入真值。

| 四种子，每模型 48 episodes | 原始 SR/CR/Timeout | 隐藏状态真值 SR/CR/Timeout |
| --- | --- | --- |
| V6 物理 KDA | 83.33/6.25/10.42% | 83.33/4.17/12.50% |
| V6 物理 GRU | 91.67/2.08/6.25% | 93.75/2.08/4.17% |

KDA 四种子 SR 差值全部为 0；GRU 只多成功一个 episode。KDA 是少一次碰撞、同时多一次超时，不是净成功提升。

这只是小样本冻结消费者诊断，不是完美训练模型或完整未来信息的上界。它目前不支持“只要把隐藏位置预测准，KDA 就会大赢”的解释。

另外，到期轨迹真值恢复和简单延长 CV 的主 SR 增益分别只有 +0.39/+0.78 pp，暂不支持扩建复杂轨迹生命周期模块。

### 7.5 训练真实观测与选动作 CV 查询不同，但尚未证明它是主要失败原因

另收集 64 条未参与训练的合法专家轨迹，cases 从 13000 开始。使用最终 IL 权重，对 762 个均匀转移比较同一真实前缀、同一专家 continuation target，仅改变末帧 query。

| 四种子平均 MSE | 实际下一观测 | CV 下一状态 | CV 位置不变、仅重置可见人 age 的 shadow |
| --- | ---: | ---: | ---: |
| 上下文 GRU | 0.00230110 | 0.00231515 | 0.00232848 |
| 物理 GRU | 0.00219292 | 0.00219781 | 0.00220714 |
| 物理 KDA | 0.00240531 | 0.00241053 | 0.00241885 |

没有出现大的平均误差恶化。age 重置不是合法修复，只是诊断；没有把它偷偷送到正式测试里。

这项检查限于专家轨迹和 IL 权重，不覆盖在线 RL 的失败状态或 20 人动作排序。因此不能说接口完全无问题，但也不能把它当成已经发现的根因。

### 7.6 实现和算子检查

- compact KDA 的 recurrence、梯度和无短卷积 FLA 参考层进行了对照。
- 候选 query 不修改真实记忆，actor permutation、mask、re-entry、checkpoint 重载都有测试。
- 最新本地 86 个 tests：83 通过，3 个可选旧资产测试跳过。
- 这不是完整 Kimi Linear 模型：没有原语言模型的 short convolution、hybrid attention 或大解码器。不能拿当前数字说完整 KDA 架构不行。
- 早期 attention 后不同 actor 的 q/k 接近，只能提示表示相似；每个人的矩阵独立，不能说其记忆被混在同一个矩阵里。这一强解释需要纠正。
- CUDA packed GRU 的 float32/TF32 与逐步算法不是逐位相等；语义和高精度检查分开处理，不能宣称跨设备所有动作必然一致。

## 8. V8 当前正在做什么

**不是新算法轮，而是学习预算诊断：所有四臂从 1,000 统一提高到 3,000 在线回合。**

四臂为：上下文 GRU、物理 GRU、物理 KDA、同容量零真实历史 KDA。数据、IL epochs、架构、奖励、动作、观测、评估都不同时变化。完整匹配的最终 IL 权重和原 IL 日志可以复用，但在线 replay 重建，运行完整的 3,000 回合，不是缺失 buffer 的续训。

| 种子 | 上下文 GRU | 物理 GRU | 物理 KDA | 零历史 KDA | 设备 |
| --- | --- | --- | --- | --- | --- |
| 419 | RL 2103 | RL 1862 | RL 1164 | RL 2554 | 4090 |
| 443 | RL 1806 | RL 2198 | RL 1464 | RL 2650 | 4090 |
| 467 | 训练和评估完成 | 训练和评估完成 | RL 269 | RL 471 | 3060 |
| 491 | 排队 | 排队 | 排队 | 排队 | 4090 |

数字是 10:16 前的同步快照，不是实时 dashboard。未完成四种子前，不发布单个种子的“胜出”。

3,000 也不叫已经收敛。原 camrl 的完整配置预算更大，本轮主要补足探索衰减结束后的学习，不能用最小训练替代论文级充分训练。

另一个重要限制：当前 V8 各臂使用统一冻结实现，但旧上下文 GRU 的 packed kernel 实现与新版本存在细微差别，导致在线轨迹可能分叉。旧 1,000 对新 3,000 的对比不是完全严格的单预算因果实验；**本轮四臂内部比较仍保持同一实现。** 有合格信号后，fresh confirmation 必须在同一设备从 IL 开始重训各臂。

## 9. 三个设备、代码和产物在哪里

### 本地台式机 3060

运行种子 467；主代码及全部回收数据保存在：

/home/abc/workspace/shixu

### 4090 服务器

运行 419/443/491，8 个并发 worker。此次文件只放 RAM 路径：

/dev/shm/shixu_occlusion_v8_20261004

没有为这轮下载大模型。系统盘剩余约 8.9 GB，当前 RAM 工作区占用很小。V6/V7 的临时工作区已在产物回收和 checksum 核对后删除，V8 完成后也采用同一清理规则。

### 笔记本

运行 CPU 诊断、冻结闭环干预和独立专家轨迹采集，不占 GPU 训练队列：

/home/jasper/shixu_occlusion_v6_analysis

### Git

https://github.com/jinglongjiang/shixu

报告写作前最新已推送提交：84a1509。V8 的冻结代码在 dd5dcb9；后续诊断改动没有修改正在训练的模型或训练代码。

Git 保存源码、协议和结果说明，不上传大量权重/轨迹文件。原始记录保存在本地以下目录：

| 内容 | 路径 |
| --- | --- |
| 早期机制 pilot | /home/abc/workspace/shixu/outputs/memory_pilot |
| 静态/动态缩放四种子 | /home/abc/workspace/shixu/outputs/scale_followup |
| 正式遮挡 V1–V8 | /home/abc/workspace/shixu/outputs/occlusion_v1 到 occlusion_v8 |
| 新接口和真值 shadow | /home/abc/workspace/shixu/outputs/query_contract |

各轮 result.json、model.pt、learning.jsonl、数据 hash 和 source hash 可追溯。V6/V7 所有远端产物已校验回收；V8 的中途同步文件尚不算最终封账。

## 10. 有没有走偏：我的判断

### 没有偏离的部分

保留了本地母体、5 人训练、5/10/20 人测试、实际权重、统一基线和清洁的单记忆结构，没有换成 Bayesian/PPO/大视觉系统，也没有把“成功运行”当作方法成功。

### 已经需要纠正的部分

1. **试结构的速度快于定位问题的深度。** V1–V7 主要是消费/写入/读取的局部替换，还没有建立强的、可重复的“同样当前观测下，特定历史会带来更好动作”的遮挡证据链。
2. **不能继续围着 gate。** 四种子已经不支持其稳定增益，且 explicit motion evidence 缺少运行时作用证据。当前工作转向实际记忆内容、读取、预算和 headroom。
3. **不能预设遮挡记忆一定能明显赢。** 当前 CV 和速度观测很强，行人又不响应机器人；需要承认 benchmark 的可利用空间可能有限，而不是默认 KDA 再调一点就会成功。
4. **近邻已有的机制不能冒充贡献。** actor recurrent memory、delta 更新、read/write 分离、task-conditioned query、时间感知递推都有先例。现在没有完成可发表方法的新颖性证明。

### 现在不能下的结论

- 不能说时序方向错了，或者 KDA 整个家族无效。
- 不能说已找到 stale memory 是性能下降的根因。
- 不能说新门控学会了选择性遗忘。
- 不能说 hidden truth shadow 就是完整最优上界。
- 不能说增加预算一定会成功。
- 不能说已经有论文级泛化优势。

## 11. 接下来如何继续，而不是盲目循环

1. 跑完当前 V8，回收并核查全部最终权重、日志、16 个结果和相同测试 keys，再给四种子总体比较。
2. 如果满足开发规则，冻结唯一版本，在未见种子/未见 cases、同一设备上从 IL 开始做确认；不在开发集上继续挑版本。
3. 如果 V8 仍不赢，不立即开始另一批 gate 或超参数。先在已保存失败 episodes 中确定：失败是否发生在合法可见信息不足、遮挡轨迹错误、候选排序错误，还是安全—进度偏置；区分可被历史补足与历史根本无法知道的事件。
4. 只有定位到一个具体缺失能力后，做一个有依据的替换。不能把“需要再训久一点”“换个大网络”“加个预测头”当成默认解释。

**当前最终状态：没有合格 winner；V8 正在完成；有可复查的训练与诊断链，但核心方法贡献仍未成立。**

这份报告的目的不是给负结果找理由，而是让你能检查：我到底改了什么、证据支持到哪里、哪些解释已经应该放弃，以及接下来是否仍在围绕真实遮挡问题推进。

## 12. 写报告后的追加诊断：失败不全在遮挡信息

回放 V6 物理 GRU、物理 KDA、零历史 KDA 的全部已存碰撞 controls，共 17/14/25 次。终端和 scene 都复现；碰撞对象在碰撞时全部可见，而且至少已连续可见 2–3 个以上控制步。三臂在最后一步都没有达到原 0.2 m endpoint clearance 的候选动作。

这不是证明遮挡没有早期影响，而是说明：**不能把碰撞终点归因于当前没有看到那个行人；陷入困境可能发生在更早的动作选择。** 需要从进入困境前的状态检查，而不是只看碰撞瞬间。

另外回放三臂全部 31/59/26 个 timeout，不改变已存执行动作，核对相同终点与路径：

| timeout 最后 10 秒的 episode 内平均 | 物理 GRU | 物理 KDA | 零历史 KDA |
| --- | ---: | ---: | ---: |
| 有原 CV 安全且向目标前进候选的帧比例 | 65.08% | 68.69% | 65.38% |
| 无原 CV 安全候选的帧比例 | 0% | 0% | 0% |
| 距机器人 2 m 内有隐藏行人的帧比例 | 3.06% | 2.25% | 0% |
| 最后 10 秒净前进，均值 | 0.039 m | -0.565 m | -0.825 m |

KDA 的 59 次 timeout 中，47 次最后 10 秒净前进绝对值不到 0.02 m，56 次这段时间附近没有隐藏行人。不是所有超时都发生在“被遮挡的人堵住路、没有动作可走”的状态。

这里的安全只是原单步 CV endpoint 规则，不是完整未来安全保证；样本只包含各自的失败轨迹，也不能据此建立遮挡或记忆的因果结论。但它把下一步诊断收紧到**合法信息下的动作排序/进度停滞**，而不是默认再增加遮挡预测模块。

不会通过偷偷加进度奖励、目标吸引力 fallback 或改动作空间来挽救 KDA 数字。那样可能解决通用导航问题，却不能证明我们的时序机制解决了遮挡问题。

原始结果：outputs/query_contract/collision_classes.json 和 timeout_replay.json；只读工具：experiments/failure_probe.py。该工具发生过 JSON 导出类型错误，修复后重新完整回放；错误运行没有产生新的训练或方法结论。

另一项小诊断只将测试 epsilon 从 0 改为已有训练末值 0.05，仍固定权重、输入、奖励和 smoothing。每臂四种子共 48 episodes：GRU 的 SR/CR/timeout 仍为 91.67/2.08/6.25%，KDA 仍为 83.33/6.25/10.42%。KDA 的配对 SR 改变为 -8.33/+8.33/0/0 pp。**少量探索没有解决总体差距，不能据此把训练探索与贪心测试差异认定为根因。** 随机动作可能绕过原贪心安全筛选，这也不是部署改法或正式性能比较。原始结果为 outputs/query_contract/behavior_shadow.json。

## 13. V8 最终封账与本轮停止

前面的 10:16 中间快照保留原样；本节取代其中“V8 正在运行”的当前状态。V8 的 16 个模型已经全部完成，最终权重重新加载后测试，共 1,536 episodes。正式遮挡 V1–V8 累计 104 个训练运行、9,984 个最终权重测试 episodes；早期全可见实验不混入这个数量。

| V8，四种子、每模型 96 测试 episodes | 总体 SR % | CR % | Timeout % | 5 人 SR % | 10 人 SR % | 20 人 SR % | 主指标 10/20 人 SR % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Context GRU | 85.42 | 6.51 | 8.07 | 93.75 | 84.38 | 78.12 | 81.25 |
| 同位置物理 GRU | 86.98 | 6.25 | 6.77 | 98.44 | 86.72 | 75.78 | 81.25 |
| 物理 KDA | 86.98 | 5.99 | 7.03 | 98.44 | 87.50 | 75.00 | 81.25 |
| 零 committed-history KDA | 87.76 | 7.55 | 4.69 | 94.53 | 92.19 | 76.56 | 84.38 |

零历史臂仍有合法 tracker 的 CV、age/gap 和当前 query，不等于无信息传感器。

### 配对证据，而不是只看平均数

种子顺序为 419/443/467/491。KDA 主指标相对 Context GRU 的差为 -9.38/-7.81/-6.25/+23.44 pp，只有 1/4 为正；相对同位置物理 GRU 为 -4.69/-6.25/+6.25/+4.69 pp，2/4 为正。两项均值为零不是统计等效性证明。

相对零历史臂，KDA 主指标低 3.125 pp，仅 1/4 种子为正；主指标 cohort 的 collision 低 1.95 pp，但 timeout 高 5.08 pp。没有满足预先固定的收益、配对方向和安全/进度规则，**没有 METHOD_ENTRY_FOUND，也没有获得 fresh confirmation 资格。**

V8 的 KDA 相对 V6 同结构 RL1000，主指标提高 4.69 pp，四种子改变为 -4.69/-1.56/+20.31/+4.69 pp。总体成功数 311→334、碰撞 14→23、超时 59→27：预算缓解了一部分停滞，同时碰撞增加；不能只报成功率增长，也不能把两个总体计数解释成逐 episode 的因果转换。

### 可复现性与计算记录

16 个模型均为 128 个相同 IL demonstrations、50 IL epochs、3,000 新在线 MC episodes。完成日志、有限权重、最终 checkpoint hash、数据/source hash、相同 96 episode keys 均核查通过。四个 KDA 的前 1,000 RL 日志与 V6 完全一致；旧 Context GRU 的 packed-GRU 实现有浮点路径差异，所以它的跨轮差值不能仅归因于预算。

95 个远端文件经 SHA256 核对后回收到本地，4090 的本轮 RAM 临时目录已经删除。原始数据和权重保留在 outputs/occlusion_v8；核查记录为 artifact_audit.json。未使用预留 fresh seeds/cases。当前测试为 88 项，85 通过，3 项缺少可选 legacy 资产而明确跳过。

KDA 参数 268,873，物理 GRU 301,313，Context GRU 333,313；KDA 每 actor 的两层矩阵状态为 8,192 个数，物理 GRU 为 256 个数。少参数不等于少运行成本。KDA 四种子的记录训练时间约 3,780–5,031 秒，物理 GRU约 1,497–3,003 秒；推理中位数分别约 9.54–14.17 ms 与 4.04–10.47 ms。设备与并发负载不同，这些仅是运行描述，不是隔离条件下的速度结论。

### 决策

**V8 追平了两组 GRU 的汇总主指标，但没有稳定超越，不能作为已有方法优势。当前版本停止，不启动 KDA V9/V10，不做 fresh confirmation，不改 reward/action/filter 来挽救数字。**

这不是 KDA 家族的 HARD_REJECT。它是对当前“短物理历史→矩阵记忆→标量价值”任务组织方式的封账：尚未证明其检索能力是必要的，也尚未定位记忆读取与价值排序之间的因果瓶颈。后续只能先补问题和证据，不继续结构试错。

## 14. 五个问题的证据审计

本节分开使用三个标签：**已证实事实**是代码/存档/指定实验直接给出的结果；**有证据支持的推断**是仍有替代解释的判断；**尚未证明的猜测**不能作为继续开发的已知前提。旧结果与 V8 不混池，动作敏感性不冒充正确性，冻结 shadow 不冒充完整最优上界。

### 14.1 有没有给 KDA 适合它的输入和任务？

**已证实事实：目前 V8 的算子和输入如下。**

每个 actor 的物理输入为 8 维：世界位置 x/y、世界速度 vx/vy、speed、radius、合法测量间隔秒数、active。先经 Linear(8→128)+ReLU；每层再 LayerNorm 后生成下列量。第二层接收第一层输出的残差特征，而不是再次输入原始八个数。

| 量 | 实际代码生成方式 | 能确认的算子角色 | 尚不能确认的导航语义 |
| --- | --- | --- | --- |
| q | L2-normalize(SiLU(Wq x))，4 个 head | 从矩阵读取内容的 learned query | 没证明它在查询意图、让行经历、有效证据或特定历史片段 |
| k | L2-normalize(SiLU(Wk x)) | 每次更新/纠正的 learned address | 没有显式事件标签或历史内容寻址监督；不能称为“意图地址” |
| v | SiLU(Wv x) | 写入/纠正的 learned content | 由物理特征学出的表示，不是已验证的 motion/context 语义槽 |
| decay | -exp(log_rate)×softplus(low-rank forget(x)+dt_bias)×interval | 各 key channel 的保留/衰减 | V8 每个合法测量写入乘 1；不是已证明的证据失效判别。V6 elapsed 另乘真实间隔 |
| beta | sigmoid(Wbeta x)，每 head 一个标量 | 对 v−kᵀS 的 delta 纠正强度 | 没证明是置信度、改向程度或可信度 |

未测量帧不写入矩阵。每个 actor 有独立状态，网络参数共享；身份键负责对齐，不是 learned key 的数字输入。两层四头 32×32 矩阵，每 actor 共 8,192 个数；同位置 GRU 状态为 256 个数。

**V8 候选动作并不直接进入 q。** encode_history 保存测得的物理历史；read_history 用第一候选的 CV 行人物理后继生成同一份逐人 query/read，再广播给 80 个候选。候选机器人位置、速度、目标和人机几何在之后的 fusion/value 阶段进入。这里的 successor 包含私有的 decay+delta 更新代数，但没有把假想证据写回真实历史。V7 才真正测试了候选条件的 q：先做人机融合，再查询同一份物理记忆。

每次评分重建最近 23 个前缀帧，不是 episode-long 外部记忆。当前输入组织主要是短时连续位置/速度及测量有效性；模型可以学出复杂函数，但没有保存或验证“事件内容—检索地址—决策用途”的显式结构。物理 KDA 相邻帧 key cosine 约 0.992，这支持地址随平滑运动高度相似，**不证明这种相似本身有害**。

**有证据支持的推断：** 当前任务更接近短时运动摘要和当前几何评分，尚未显示需要超出 GRU 的大量独立关联检索。V8 KDA 没有胜出、零历史模型也很强，都不支持矩阵记忆“有必要”的主张；但这不是矩阵记忆能力不足的证明。

V1–V7 的 context、local address、候选 query 均真实实现并训练；它们没有形成优势的**根因仍不知道**。V4 改变地址没有改善结果，削弱了“只要把 address 分开就能解决”；V7 query 版本输健康同位置 GRU，削弱了“候选寻址自然带来优势”。这些反例不能替代为什么失败的因果解释。

**尚未证明的猜测：** q/k/v 是否具有适合该任务的语义、关联内容是否可检索、读出的内容是否是价值头需要的，全部尚未证明。当前也没有实验说明这些内容为什么不能由 GRU hidden state 表达。

**结论：不能在“KDA 没能力”和“信息组织不合适”之间作因果裁决。能确定的是：我们没有证明自己构造了需要且能利用 KDA 关联检索的任务/表示。因此不应把当前负结果归因于 KDA 家族无能力，也不应把改输入必胜当作已知答案。**

代码来源：shixu/motion.py 的 physical_features、MotionKDACell.parameters_for、successor/read、read_history；shixu/temporal.py 的 delta_scan。

### 14.2 遮挡环境是不是太简单，CV/GRU 是否吃掉 headroom？

**已证实事实：** 有真实遮挡和重现，但当前传感器给可见行人的瞬时真速度、理想已见身份关联；tracker 自带两秒 CV；行人不响应机器人；窗口约六秒。128 条自然专家轨迹中隐藏位置误差均值 7.65 cm、近 2 m 均值 2.72 cm。它们是预测统计，不是决策 sufficiency 统计。

| 用户要求的数量 | 当前能给的结果 | 不能偷换成什么 |
| --- | --- | --- |
| current/CV 足够、历史无额外正确决策价值的状态比例 | **证据不足，未测得** | 隐藏记忆屏蔽后 259/338 动作不变，不等于这些状态历史没价值 |
| 历史显著改善正确候选排序的比例 | **证据不足，未测得** | 79/338 隐藏记忆敏感、344/626 全历史敏感，只是动作变化 |
| 真正因遮挡期间信息缺失而失败的比例 | **证据不足，未建立配对因果标签** | V6 碰撞时对象全部可见，不排除此前受遮挡影响 |
| 补准隐藏当前状态后的闭环收益 | V6 小 cohort：KDA 40/48 成功保持不变；GRU 44/48→45/48 | 不是全部未见人/未来轨迹/理想消费者的 Oracle |

V6 全 timeout 回放显示 KDA 59 次中 56 次最后十秒附近没有隐藏人，47 次最后十秒几乎无净进展；其最后十秒约 68.69% 的帧仍有原 CV 安全推进候选。这说明大量停滞不能直接解释为“此时缺少隐藏人位置”，但不能把它们全部判为与历史无关。

为什么隐藏真值几乎不提高 SR？**尚未证明。** 较小当前误差、冻结消费者不会利用修正、未来仍未知、碰撞减少同时等待增加，都与现有结果相容。只替换 retained-hidden 当前状态而不重训消费者，也不修复历史，不是完整上界。

V8 主 cohort 中，物理 GRU 成功而 KDA 失败 26 个 episodes，反向也有 26 个，182 个共同成功。同一场景 outcome 不等于同一路径；没有做 GRU 的历史配对反事实，**不能说这 26 个 GRU 优势来自历史，而非更合适的 value/ranking**。

**有证据支持的推断：** 这套 benchmark 的当前输入/CV 已经较强，而“需要更复杂时序模型才能利用”的 headroom 很可能小于整体失败率所暗示的空间。V8 零历史主 SR 84.38%，有历史 KDA 与两个 GRU 都 81.25%，进一步削弱了默认需要更复杂记忆的前提。

**尚未证明的猜测：** benchmark 完全没有时序 headroom、GRU 已达上界、所有 failure 都是 value head 问题，均不能成立。20 人仍有失败并不证明失败可由时序信息补足。

**结论：目前没有证据证明这里存在足够大、CV/GRU 不能解决的 temporal headroom。也没有合法 Oracle 证明其不存在。工程上已有任务；科学上尚未合格到值得继续盲做 KDA 的程度。**

### 14.3 损失发生在决策链哪一层？

补做只读诊断，不训练、不改消费者：先按 archive key 固定每种子、每 source arm 的首个 10/20 人碰撞和超时，各 8 条；在终点前 10/8/6/4/2 秒取状态。短于该时间的自然轨迹不补造前史，最终 8 条 timeout 有 40 个状态，6 条 collision 有 15 个状态。GRU/KDA 的 IL50、RL1000、RL3000 对同一合法前缀和 80 候选评分。

真实参考为已回放行人轨迹、候选机器人速度保持两秒、分段连续几何距离。机器人对行人不可见，故该离线 shadow 的行人轨迹不因替换候选而改变；全部真值只用于评估。它回答短期安全/进度，**不是最优 action ranking，也不是闭环换动作后的成功率**。源码/记录为 experiments/ranking_audit.py 和 outputs/query_contract/closure_ranking_audit.json。

**已证实事实：** 55/55 源模型的计算执行命令与存档一致；每次手工拆开的过滤评分也与原 policy.score 数值一致。没有额外训练或改变原始正式结果。首次运行误用了全局旧 crowd_sim，未产生诊断结果；改用已冻结 vendor 路径并加入路径检查后完整重跑。

| 决策链位置 | 现在能定位的事实 | 未能定位的内容 |
| --- | --- | --- |
| raw observation | 同一状态、actor ID/mask/age、前缀、80 候选完全相同 | 不存在只给 KDA 较差传感器的混杂 |
| actor feature | 同样八维物理输入，但编码器分别训练 | learned embedding 数值不同不等于某个编码器错误 |
| temporal state | GRU vector 与 KDA matrix 分别由合法前缀生成 | 不同 latent basis/shape 没有可比较的正确答案 |
| memory read/query | 当前 query 合同和无写回由测试确认；屏蔽历史会改变评分 | 没有 read-level 语义真值或共享价值头反事实，无法断定 read 已错 |
| current/history fusion | 同位置 post-memory fusion 和 pooling/head 结构相同、权重不同 | 尚未分开归因 memory 与训练后的 fusion/head |
| scalar value→native ranking | 超时前 KDA 更少把单步安全推进动作排到第一 | 不能把每个短期推进动作称为最优动作 |
| safety/risk filter | 使用相同候选 clearance，部分状态救回安全推进动作 | 没证明它全面导致或全面消除差距 |
| smoothing→execution | 同样 alpha=0.3 和同一历史执行命令；可以改变短期进度 | 尚未做完整关闭 smoothing 的配对闭环，不能定为超时根因 |

RL3000 的共同 timeout 状态中，32/40 有单步 CV clearance≥0.2 m、向目标推进≥0.02 m 的候选。原 r+gamma V 排第一的 GRU 为 20/32，KDA 为 8/32；过滤后为 27/32 与 15/32。最好的这类候选原评分排名中位数分别 1 与 8，过滤后 1 与 2。**这部分差异已在过滤之前存在；过滤不是全部差异的起点。**

进一步拆开 scalar V 与 immediate reward：同一 32 个状态中，直接按 V 排第一的安全推进选择为 GRU 15/32、KDA 3/32，最好安全推进候选的 V 排名中位数为 2 与 8。这把差异推进到**模型输出价值时已存在**，不是只在加 risk penalty 后出现；仍无法分开归因记忆与价值头。

| 同一 32 个有 CV 安全推进候选的 timeout 状态 | GRU 选择安全推进 | KDA 选择安全推进 |
| --- | ---: | ---: |
| 只按 scalar V 排序 | 15 | 3 |
| immediate reward + gamma V | 20 | 8 |
| 原 safety/risk filter 后 | 27 | 15 |
| 同一之前执行命令，原 smoothing 后 | 20 | 13 |

平滑使 GRU 失去 7 个单步合格选择、KDA 失去 2 个；在这组状态它并未扩大 KDA 与 GRU 的数量差距。这里“合格”只针对明确的单步阈值，不是整个导航最优策略。

在 KDA 自己的四条 timeout 路径上，终点前 10/8/6/4/2 秒最优 CV 安全推进候选的原评分排名分别为：seed419 的 27/9/1/2/2；443 的 1/3/2/1/3；467 的 55/55/55/55/55；491 的 8/8/8/8/8。不是所有超时都同一机制，不能只挑 seed467 做普遍归因。

**反例必须保留：** 更严格的“两秒真实间距≥0.2 m 且净推进≥0.16 m”候选在 20/40 状态存在，GRU 原排序选中 5/20、KDA 8/20，过滤后仍为 5/20 与 8/20。因此“单步进度排序低”不等于“完整短期决策全面更错”；部分是安全—进度偏好。较宽松的真实不碰撞推进集在 40/40 存在，原排序 GRU 15/40、KDA 13/40。

两秒严格参考中，timeout 的 smoothing 使 GRU 合格动作从 5 个增到 10 个，KDA 8 个保持不变；collision 状态中 GRU 4→3、KDA 3→3。样本有限，但不支持“smoothing 普遍只伤害 KDA”的说法。

collision 的 15 个状态中，严格两秒安全推进集在 12 个存在；KDA 最好候选原评分排名中位数 9.5，GRU 3.5。两秒不碰撞推进集的对应中位数为 10 与 3。**在部分碰撞前状态可以看到上游排序偏差，但不能归因 read/head，也不能证明提前五到十秒逐步恶化：十秒前只有 1 个、八秒/六秒前各 2 个状态，轨迹并非一致单调。**

**有证据支持的推断：** 原生价值/动作排序的安全—进度配置是实际停滞的一个近端问题；单靠加入隐藏状态预测或调整最后过滤，不能保证解决它。

**尚未证明的猜测：** 首个错误具体是 q/k/v、memory read、fusion 还是 value head，目前证据不足。当前定位到的是“部分异常已出现在 native ranking”，不是第一次错误的因果定位。此前若说已查明 stale memory/read 是根因，应撤回。

### 14.4 MC 训练目标有没有驱动时序检索？

**已证实事实：** 目标为历史窗口到 scalar MC return 的 MSE；部署用 80 候选的 r+gamma V 排序。没有 retrieval label、候选间排序监督或读地址语义监督。目标未显式约束检索不等于它不能间接学出检索。

正式遮挡轮保存 IL50 与最终 RL1000/V8 RL3000；早期 scale_followup 的 RL500 确实存在，但它是全可见 aligned 输入、另一组种子和另一个模型族。不能把它补在正式遮挡曲线中。训练日志每回合有 loss，但没有存完整共享在线 replay；不同模型自己的在线轨迹/return 不相同。

同一份未参与训练的 64 条专家轨迹，762 个相同合法转移、相同专家 continuation targets，三阶段冻结权重误差如下：

| 阶段 | 物理 GRU，四种子平均 | 物理 KDA，四种子平均 | 正确解释 |
| --- | ---: | ---: | --- |
| IL50 | 0.00219292 | 0.00240531 | 同一专家目标的 held-out MSE；KDA 均值高约 9.7%，3/4 种子更差 |
| RL1000 | 0.11450898 | 0.11557634 | 对专家 return 的偏离，不是 learned policy 自己的真实 value error |
| RL3000 | 0.11158965 | 0.12009061 | 同上；也不是在线 replay 的训练/验证 MSE |

IL 拟合已有轻微差异，不能说差距一定始于在线 MC；但没有 IL checkpoint 的正式闭环测试，**不能把 IL MSE 较高说成 IL 导航已显著落后**。

在同一 40 个 retrospective timeout 状态上，IL50 两模型的原排序各有 15/32 选中 CV 安全推进；到 RL3000 是 GRU 20/32、KDA 8/32。这说明这些冻结状态的功能分化在在线学习后更明显，但状态是最终失败条件下选出的，不能当作全分布学习曲线或 MC 目标因果证据。

**有证据支持的推断：** 平均专家拟合误差不能替代 action ranking/闭环质量；目标与部署用途间有值得检查的差异，预算对 KDA 也确有部分影响。

**尚未证明的猜测：** “MC 目标根本不适合 KDA”“KDA 特有 good-MSE/bad-ranking 现象”“只加 ranking loss 就能修复”均未证明。当前缺少同状态 80 动作的正确 continuation targets、两模型的固定共享 on-policy replay 和相同目标下的配对误差；专家分布 MSE 与 failure-state 排名还不是同一分布，不能计算并宣称可靠相关性。

**结论：训练目标是候选解释，不是已查明瓶颈。本轮不能以此启动改目标的新训练。**

### 14.5 是否偏离目标，以及 stop/go

**已证实事实：** 原目标是找到真实时序信息缺口、在 5 人训练后稳定泛化，不是强制 KDA 获胜。以下表重新审计版本假设，失败版本仍完整保留：

| 版本 | 当时想验证的假设 | 当前判断 | 增加的理解/局限 |
| --- | --- | --- | --- |
| V1 | 合法测量写入和 hidden 读取能改善遮挡导航 | 性能假设削弱；具体历史价值未知 | 合同跑通，但没证明缺的历史能改善正确动作 |
| V2 | 改观测读取时钟可解决 query 消费失配 | 具体修复削弱 | 主要是局部接口尝试，不是正确读取的证据 |
| V3 | 历史邻居 context 能让 KDA 更好处理遮挡 | KDA 收益削弱；context 的普遍价值仍未知 | 健康 context GRU 很强，不能把上下文判为无用 |
| V4 | 局部 q/k address 可消除 context 地址混淆损失 | 强根因解释削弱 | 地址统计可改，收益不随之出现；独立 actor 矩阵从未真的共享混写 |
| V5 | 私有 successor update 弥补 read-only 的缺陷 | 具体性能假设削弱 | 支持不写回的实现合同，未证明哪一种决策含义正确 |
| V6 | 物理历史与候选几何分离、elapsed decay 有增量价值 | elapsed 稳定收益削弱；表示适配仍未知 | 同位置 GRU/零历史很强，不能用弱控制救 KDA |
| V7 | 候选动作条件查询能取得检索优势 | 具体实现收益削弱 | readonly 比弱 branch 好，不是胜过强基线 |
| V8 | RL1000 预算不足解释主要差距 | 部分预算效应有支持；充分解释削弱 | KDA 汇总追平，但种子不稳定、碰撞增加、没过门槛 |

V2、V5 和 V6 的时钟变体主要是局部结构/接口尝试，没有建立新的 decision-relevant history 事实。V3/V4/V7 缩小了候选解释，但没有找到可发表能力。V8 是有依据的预算检查，不是新算法贡献。

**有证据支持的推断：** 已经存在“为救 KDA 持续换接口”的明显风险。问题不是试八版本身，而是没有先证明所组织的内容值得检索，就用结构合理性安排下一版。我此前在这个顺序上推进过快，不能继续用新的合理故事圆旧负结果。

**尚未证明的猜测：** KDA 永远不适合导航、时序方向没有价值，均不成立。但类似先例和模型热度也不是继续投入的充分理由。

**STOP：当前停止 V9/V10。** V8 达到两组 GRU 的汇总主指标而不是稳定超越，尚未证明 KDA 所需的不可被强简单机制吸收的能力；本轮结束，不继续训练，不把参数、reward 或测试场景调到赢。

**重新 GO 的最低证据：**

1. 自然任务里存在 current/CV 近似相同、合法历史不同、合理动作/后果也不同的可重复 cohort；离线 intervention 显示有明确可利用收益，而不只是 action sensitivity。
2. 能指出需要检索的历史内容、其来源、地址和当前决策用途；例如非局部事件/长期交互证据的具体恢复，不能只有平滑位置速度再包一层矩阵。
3. 在健康 parent 和足够训练下，普通 actor GRU、短窗和 CV 的剩余能力缺口仍对应这个问题；不是提前要求它们必须失败，而是正式比较后有 residual。
4. 新输入/机制能由合法观测生成，消融可以把收益归到这一个能力，fresh paired seeds/cases 能确认，无重大安全—进度牺牲。

若停止，下一步先寻找**真实感知条件下 current state 无法直接给出的时序信息**：从历史估计运动、遮挡下的动态行为变化、持续交互证据或自然身份连续性。选择依据应是数据中的 action/outcome headroom，而不是哪个场景最容易让 KDA 赢；当前理想速度/身份与非响应人群的边界须重新说明。先决定 representation 是否确实需要内容寻址，再决定是否用 KDA 或其他时序模型。这里是下一研究资格，不是已批准的新训练任务。

## 15. 本轮交付与停止记录

V8 最终权重、学习日志、episode records、冻结协议和完整核查留存；主报告、README、策略报告与候选状态报告均更新。追加诊断不改变正式数字，新增测试覆盖连续距离、排名和原 evaluator 数值一致性；当前 91 tests 中 88 通过，3 个可选 legacy 资产跳过。主模型/奖励/模拟器的冻结 source hash 未改变。

**本轮状态：完成 V8、完成已有证据的五问审计；没有新算法通过；停止当前 KDA 版本循环。没有后台训练、没有 V9/V10、没有 fresh confirmation。**

<!-- END PRESERVED SOURCE -->


---

<a id="stage-2"></a>

## Source: README.md (historical README section 1)

Original full-source SHA-256: 0f5332ec89c79e6ab9fc605bb8e502e3aca61323ac48851518db15cb3f732d9a

<!-- BEGIN PRESERVED SOURCE -->
# shixu

A small temporal crowd-navigation research framework extracted from the user's
local camrl Mamba-VL project. It contains the cleaned parent and explicit
temporal architecture experiments, not an established new-method claim.

## One Main Path

```text
legal observations -> frame encoder -> temporal encoder -> scalar value
                    -> original candidate successor/value lookahead -> action

ORCA demonstrations -> Monte Carlo value initialization -> online MC refinement
```

GRU is the default temporal baseline. Mamba is optional and retained only for
legacy comparison. There are no Double-Q/PPO/SAC branches, auxiliary prediction
heads, Bayesian modules, teacher networks at deployment, or fallback backbones.

The explicit actor-first alternative moves temporal encoding before crowd pooling:

```text
identity-bound human histories -> shared temporal encoder -> crowd pooling
                              -> scalar value -> unchanged lookahead
```

The processing-order prototype now compares scene-first and actor-first GRU
using identical aligned observations and exactly the same parameters. Every
human uses the same GRU weights, with independent histories. That trial adds no
dual-memory system, contradiction detector or additional prediction head.
That processing-order experiment remains a baseline, not a selective-revision method.

The observation-only contract consumes robot raw state9 and human observed
motion9 plus presence. It removes redundant legacy relation features rather
than adding a predictor or another memory. The explicit legacy feature contract
remains only for reproducing the first trial and loading its checkpoints.

## Layout

| File | Responsibility |
| --- | --- |
| shixu/features.py | Legacy observation contract and history windows |
| shixu/observations.py | Episode-local observed association keys, never numeric ID features |
| shixu/model.py | Frame encoder, replaceable temporal encoder, value head |
| shixu/temporal.py | Compact GRU/KDA/GDN2 actor memories; real-write/candidate-read interface |
| shixu/policy.py | Original action support and successor-value evaluation |
| shixu/replay.py | Episode-safe windows and MC targets; no duplicated window archive |
| shixu/runner.py | One runner for collection, training and evaluation |
| shixu/training.py | ORCA value initialization and online MC refinement |
| shixu/cli.py | Explicit collection/evaluation/training commands |
| vendor/crowd_sim | Frozen local simulator dependency |
| experiments/ | Frozen protocol, immutable ORCA collection and paired processing-order trial |

## Occlusion Development Loop

The new `occlusion` architecture removes the five-person input cap. It uses
episode-local identity slots, actual-measurement write masks and a separate
retained-track read mask. A previously observed actor can remain relevant while
occluded; an actor that has never been seen cannot enter the model. Missing
positions use the last legally measured velocity for at most two seconds.
Predicted positions and candidate successors are read-only queries, not new
measurements. There is one shared actor memory and no external memory gate,
auxiliary predictor, dual-memory branch or changed reward.

```text
body-occluded observations -> legal track histories -> shared KDA memory
                          -> retained-track candidate reads -> scalar value
                          -> inherited 80-action value lookahead
```

This is a functional research prototype, not an established novelty or
performance claim. The common retention interface is also used by the
current-state and recurrent comparisons. Previously reported full-observation
results are not occlusion results.

```bash
# Pin the vendored simulator when another CrowdNav is installed locally.
export PYTHONPATH=vendor:.
python -m unittest discover -s tests -v
python -m experiments.occlusion collect \
  --data outputs/occlusion_v1/demonstrations.pt
python -m experiments.occlusion queue \
  --root outputs/occlusion_v1 --data outputs/occlusion_v1/demonstrations.pt \
  --device cuda --workers 4
python -m experiments.occlusion summarize --root outputs/occlusion_v1
```

`experiments/occlusion_protocol.json` freezes shared demonstrations, four
paired development seeds, 50 IL epochs, 1,000 online MC-RL episodes and
5/10/20-person evaluation. Every reported model is reconstructed and loaded
from its final saved checkpoint before evaluation. Development cases guide
diagnosis; fresh seeds and separate confirmation cases remain reserved until
an architecture is selected. A negative version is diagnosed, not relabeled
as a failed research family. Raw weights, logs and episodes remain local.

The second frozen version, `occlusion_observation_protocol.json`, changes only
the read clock. Memory is read from the latest legal history frame once per
actor, then shared by all candidate actions. Candidate geometry remains in the
spatial value encoder, but it no longer changes the actor's temporal read vector.
The weights and parameter count are unchanged by this switch. Both versions
receive byte-identical episode observations and rewards. The current-track
reference is reused because it has no learned temporal read; the temporal
arms are retrained through the complete IL/MC-RL schedule.

```bash
python -m experiments.occlusion collect \
  --protocol experiments/occlusion_observation_protocol.json \
  --data outputs/occlusion_v2/demonstrations.pt
python -m experiments.occlusion queue \
  --protocol experiments/occlusion_observation_protocol.json \
  --root outputs/occlusion_v2 --data outputs/occlusion_v2/demonstrations.pt \
  --arms gru kda --device cuda --workers 4
python -m experiments.occlusion summarize \
  --protocol experiments/occlusion_observation_protocol.json \
  --root outputs/occlusion_v2
```

The observation-clock design is a hypothesis under test, not a correctness
repair. Both versions leave the stored matrix state unchanged during candidate
evaluation. Action-conditioned reads can legitimately retrieve different
information for different decisions even when humans do not react to the
robot. The new bias separates a shared temporal read vector from
candidate-conditioned geometry evaluation and avoids replicating each actor's
matrix memory 80 times. Frozen-weight read ablations are diagnostic, not
substitutes for retraining or proof of better navigation.

The first full four-seed occlusion trial is complete (384 held-out development
episodes per arm, not the earlier full-observation trial):

| Model | Overall SR | CR | Timeout | 10/20-person SR |
|---|---:|---:|---:|---:|
| CV-track current-value | 81.77% | 4.95% | 13.28% | 79.30% |
| Actor-GRU | 77.34% | 6.25% | 16.41% | 75.78% |
| Actor-KDA, candidate read | 78.39% | 11.20% | 10.42% | 75.78% |

KDA does not pass: primary SR is unchanged against GRU, while primary collision
increases by 5.86 percentage points. A frozen-weight switch to observation-clock
reads also worsens seed 443 overall SR from 70.83% to 56.25%; this is a
distribution-shifting diagnostic, not a trained comparison. The second version
must therefore earn its own result through matched IL and RL.

Completed final-IL checkpoints may be moved to a faster host using `--il-root`.
The experiment verifies seed/configuration and all IL log epochs, restores the
replay sampling stream, then starts a fresh, full-budget RL run. Interrupted RL
work is archived and charged separately; it is not used for checkpoint selection.
An exact CPU pipeline test checks identical full-run versus final-IL-reuse RL
updates. Different GPU/software environments can still introduce numerical
differences and are recorded with each run.

The complete observation-read trial is negative: overall SR is 63.54% for
KDA versus 79.43% for GRU; primary 10/20-person SR is 64.45% versus 76.56%.
KDA loses primary SR in all four paired seeds. Fewer collisions come with
substantially more timeouts (30.47% overall), so this version is not retained
as the candidate architecture. Shared observation reads are not an established
correction to the original action-dependent read design.

The next structural test, `occlusion_context_protocol.json`, returns to the
original candidate clock and moves the existing attention before actor memory:

```text
real actor observations -> attention (actor outputs, no scene pooling)
                        -> shared, measurement-masked KDA writes
candidate geometry     -> same attention -> read each actor memory
                        -> masked max -> scalar value -> original lookahead
```

There is still one attention and one temporal operator. The parameter counts
are unchanged. Prefix attention keys use actual measurements; hidden tracks
cannot supply fresh write evidence. Query attention can use legally retained
CV tracks. The non-temporal current-value arm is mathematically unchanged;
its existing saved weights reproduce every executed action and terminal outcome
in all 96 seed-419 development episodes after the helper refactor.

This tests whether historical neighbour context is useful. The old actor
memory is exactly insensitive to another actor's past when its own history is
fixed. The replacement removes that insensitivity in a fixture, but this is
not proof of navigation headroom or novelty. A separate episode-disjoint
linear probe of 887 natural re-entries does not improve average velocity
prediction over CV, so richer context is not presumed useful in every task.
Archived collision replays also show visible colliders and no margin-safe
action at the final step; they do not establish that occlusion caused the
earlier poor decisions. DS-RNN and [PaS](https://github.com/yejimun/PaS_CrowdNav)
already study temporal/social inference. Moving attention is an experimental
representation choice, not a standalone new-method claim.

The complete context-write trial is negative for KDA: overall SR/CR/timeout
are 62.76/12.24/25.00%, versus 85.94/5.99/8.07% for GRU. Primary 10/20-person
SR is 60.16% versus 85.55%, with KDA losses in all four paired seeds.
On 836 uniformly sampled demonstration windows, seed-443 final KDA keys have
mean inter-actor cosine 0.99926 in this version versus 0.77600 in V1.
Separate per-actor matrices do not prevent homogenization when their inputs
are almost identical. High cosine is a representation diagnostic, not a proof
that it causes all observed navigation losses.

The next test, `occlusion_address_protocol.json`, preserves local features:

```text
actor feature u_i -> u_i + attention(u, measured actors) -> memory content
actor feature u_i -------------------------------------> KDA q/k (custom)
candidate feature + same attention -> retained-memory read -> max -> value
```

All four arms share the residual spatial path. Current-value, contextual GRU,
ordinary residual KDA and actor-addressed KDA are retrained; the old current
reference is not reused because its query representation changes too. The
two KDA arms have exactly the same parameters and initialization. Only q/k
source differs; values, decay and write strength consume contextual content.
At deeper layers both streams receive the same recurrent output. There is
one attention, one shared actor memory and no additional loss or output head.
Existing read/write versions remain available for their archived checkpoints.

This is a mechanism test, not an established new method. Separate-source
delta memory already appears in [DRAM](https://arxiv.org/abs/2609.32453), and
retrieval design is studied in
[Advantage-Driven Explicit Memory](https://arxiv.org/abs/2608.25610).
The hypothesis here is narrower: retaining actor-local addresses while
remembering observed social context helps legal hidden-track action evaluation.
That claim still needs navigation gains and an occlusion-specific analysis.

```bash
python -m experiments.occlusion queue \
  --protocol experiments/occlusion_address_protocol.json \
  --root outputs/occlusion_v4 --data outputs/occlusion_v4/demonstrations.pt \
  --device cuda --workers 4
python -m experiments.occlusion summarize \
  --protocol experiments/occlusion_address_protocol.json \
  --root outputs/occlusion_v4
```

Demonstrations are the same saved observations/rewards as V1, with protocol
metadata retargeted for the matched run; there is no extra training data.
Final IL and RL weights, all failed versions, raw episode controls and consumed
compute are retained. Fresh seeds and unseen confirmation cases are not used
to develop this version.

The complete V4 result is negative. Overall SR/CR/timeout are 68.75/11.72/19.53%
for actor-addressed KDA, 72.40/10.94/16.67% for vanilla residual KDA and
77.08/5.47/17.45% for residual GRU. Primary SR falls by 11.72 pp against GRU;
actor-local addresses do not rescue this version. The final seed-443 address
probe does recover distinct keys (mean inter-actor cosine 0.8144), so loss of
actor addresses alone is not an adequate explanation of the navigation failure.

`occlusion_branch_protocol.json` tests the inherited successor-value interface,
not a new gate or a claim that private branches are novel:

```text
actual measured actor frames -> shared KDA -> real-history matrix
candidate CV successor      -> private one-step KDA calculation
                            -> existing attention/max/value -> original lookahead
```

Candidate computations never commit into real history. The single-step read
is algebraically identical to an explicit private matrix update, but avoids
80 replicated matrices. Its paired control additionally accepts legally
propagated CV tracks as history pseudomeasurements; it receives no hidden truth.
Both new arms retain identical parameters, initialization and IL/RL budgets.
The unchanged V1 current-value and read-only KDA references and the stronger V3
context-GRU reference remain controls, with configuration and action-parity
checks before reuse. Any positive development result still requires new seeds
and unseen cases, with all confirmation arms trained anew.

Engineering optimizations share matrix reads across candidates and pack measured
GRU frames without changing their recurrence. An exclusive laptop CPU read-only
benchmark falls from 5.91 to 2.37 ms for five actors and 27.67 to 9.90 ms for
twenty actors; these are not full-controller latency or navigation gains.
Frozen checkpoint actions and gradient/recurrence tests validate the changes.

## Installation

Install a PyTorch build appropriate for your machine first. Then:

```bash
git clone https://github.com/jinglongjiang/shixu.git
cd shixu
python -m pip install -e .
python -m pip install 'git+https://github.com/sybrenstuvel/Python-RVO2.git'
```

Python-RVO2 needs its normal native build prerequisites. The GRU path does not
require Mamba, Transformers or custom CUDA kernels. Legacy Mamba checkpoints
require the optional mamba-ssm 1.2.0 dependency and a matching CUDA/PyTorch wheel;
do not silently substitute another network if it fails to import.

## Commands

```bash
# Interface checks only; this is not a trained policy result.
python -m shixu.cli smoke --output outputs/smoke.json
python -m unittest discover -s tests -v

# Collect legal, identity-tagged observations and ORCA returns without training.
python -m shixu.cli collect --cases 0 1 --output data/orca.json

# Trained-checkpoint evaluation. Weights are intentionally not uploaded.
python -m shixu.cli evaluate --backbone mamba --device cuda \
  --weights /path/to/rl_model_ep10000_T24.pth --cases 0 1

# Explicitly opt into training.
python -m shixu.cli train --il-episodes 5 --rl-episodes 10 \
  --device cuda --output weights/gru.pt
```

Models must use the same config when comparing them. The optional local-source
regression tests use environment variables CAMRL_PARENT and CAMRL_CHECKPOINT;
they check features, value outputs, actions and history against the original
source. They skip explicitly when those local assets are unavailable.
New checkpoints include their model/observation configuration; evaluation uses
it automatically unless an explicit --config override is supplied.

The matched trial is driven by experiments/temporal_protocol.json, not test
results: four paired seeds, a shared 128-episode successful ORCA dataset,
50 IL epochs, 1,000 MC-RL episodes per arm, and fixed circle/square cases at
5/10/20 humans. The initial legacy contract has 302,337 parameters per arm;
the observation-only contract has 300,417 at the same width 128 and depth 2.
Only the final-budget checkpoint is evaluated. Processing-order prototype
results cannot be represented as a new algorithm or proof of selective memory.

```bash
python experiments/temporal_collect.py --output data/demonstrations.pt
python experiments/temporal_order.py run --seed 17 --order pair \
  --data data/demonstrations.pt --root outputs/temporal_v1 --device cuda
python experiments/temporal_order.py summarize --root outputs/temporal_v1

# One shared interface rescue: same data/budget, subtract legacy derived inputs.
python experiments/temporal_order.py run --seed 17 --order pair \
  --feature-contract observed --data data/demonstrations.pt \
  --root outputs/temporal_v2 --device cuda
```

Run the other seeds in the protocol before requesting the paired summary.
Native experiments assume perfect observed association and retain the original
five-human neural input cap even when the simulator contains 10/20 humans.
Missing observation masks preserve actor state; association errors and
real-world re-identification are not solved by this interface.

## Initial Matched Result

Four paired seeds completed 50 IL epochs + 1,000 online MC-RL episodes per arm,
followed by 96 fixed native evaluations each (768 total).

| Model | SR | Collision | Timeout | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Scene-first GRU | 75.52% | 9.90% | 14.58% | 302,337 |
| Actor-first GRU | 78.39% | 8.33% | 13.28% | 302,337 |

The +2.86 pp mean SR change has only 2/4 positive seed pairs and does not meet
the frozen +3 pp / 3-of-4 direction gate: NO_STABLE_GAIN. Pooled square gains
and smaller actor seed dispersion are exploratory, not a new-method claim.
Same-device RTX 3060 scoring medians are 3.110 / 4.418 ms for scene / actor;
actor-first is not a computation-saving result. No GDN/KDA/revision cell is
claimed successful on the strength of these mixed outcomes.

```bash
python -m experiments.temporal_latency --root outputs/temporal_v1 --seed 17
python -m experiments.temporal_revision_shadow --root outputs/temporal_v1 --seed 17
```

The shadow uses arrived motion evidence and native scene replay. A masked-prefix
intervention is an offline diagnostic, not a trained or deployable revision
policy. Full results/checkpoints stay local under outputs; weights and data are
not committed. The unchanged fresh follow-up used seeds 103/137: SR changes
were +7.29 / -12.50 pp, so the initial seed-dispersion signal did not replicate.
The common observation-only rescue is frozen separately in
experiments/temporal_rescue_protocol.json; its results must not be pooled with
the legacy-contract cohort.

## Completed Observation-Only Rescue

The one permitted rescue subtracts the inherited redundant/incorrect derived
inputs for **both** arms, without changing data, reward, budget or network size.
Four paired seeds again completed 50 IL epochs, 1,000 MC-RL episodes and 96
fixed evaluations per arm (768 evaluations).

| Model | SR | Collision | Timeout | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Scene-first GRU | 75.52% | 11.20% | 13.28% | 300,417 |
| Actor-first GRU | 75.00% | 6.25% | 18.75% | 300,417 |

SR differences are +3.13, +5.21, -5.21 and -5.21 pp across seeds
17/29/43/71. The mean is -0.52 pp with 2/4 positive pairs:
**NO_STABLE_GAIN** under the unchanged gate. Collision decreases in all four
pairs, but timeout increases; this is a safety-progress operating-point signal,
not proof of better navigation. Pooled 20-human gains also remain only 2/4
seed-positive. Same-device scoring medians are 3.012 / 4.163 ms (scene / actor),
so actor-first is about 38% more expensive in this workload.

The legal observed-change shadow finds four first events in 12 native
episodes: targeted actor-history truncation changes no root rankings and gives
no safe progress gain >=0.05 m. Selective-revision headroom remains unproven;
the small masked-prefix intervention does not reject the research family.
Attention and pooling both move relative to recurrence, so this comparison
does not isolate identity continuity alone.

Reserved fresh rescue seeds 191/223 were not run within that study because the
primary gate failed. That study added no GDN/KDA, new reward or extra teacher.
Across the separate initial, fresh and rescue cohorts, 20 models and 1,920
matched evaluation episodes are retained locally. None is relabeled as a new
method. All 38 local tests pass with the original comparison assets configured;
the laptop passes 35 tests with three explicit original-asset skips.

```bash
python experiments/temporal_order.py summarize --root outputs/temporal_v2
python -m experiments.temporal_latency --root outputs/temporal_v2 --seed 17
python -m experiments.temporal_revision_shadow --root outputs/temporal_v2 --seed 17
python -m shixu.cli evaluate --weights outputs/temporal_v2/17/actor/model.pt \
  --device cuda --cases 0 1
```

## Baseline Boundary

The source baseline comes from the user's
CrowdNav(20260511_last_version_mamba_vl).zip, not the later Bayesian-replaced
active camrl directory. The simulator preserves that archive's behavior;
only trailing whitespace is cleaned.
Its CrowdNav foundation is attributed in vendor/CROWDNAV_LICENSE.

The inherited deterministic baseline uses 80 moving actions, dt=0.25 s,
24-frame history, and r+0.99V lookahead. The archive's evaluation settings also
include clearance filtering, a risk penalty and action smoothing. They are
retained explicitly in shixu/default.ini; this is not a reproduction of paper
statistics based on a few episodes.

Legacy metadata indices and spatial relational-coordinate conventions are
preserved for checkpoint parity. Their audit is separate from method novelty;
changing them together with a new memory would confound that comparison.

The new training runner preserves the IL-to-MC-value-learning formulation, not
every historical launcher's behavior: observations are recorded even during
exploratory controls, teacher state is cleared between episodes, and test-case
scheduling is explicit. All new training arms must share this runner. Legacy
training numbers cannot be attributed to this cleanup without matched reruns.

Simulator IDs are association keys attached to observed states, not neural
features. Human goals/future states are not written into deployable inputs.
Weights, data, videos, credentials and old experiment artifacts are excluded
from version control.

## Explicit Memory Architecture Pilot

A separately authorized pilot compares two mechanisms without assuming the
newer operator is better:

```text
observed actor prefix -> one shared GRU/KDA/GDN2 -> per-actor state
candidate successor  -> query that state       -> current feature + memory
                     -> original attention/max pool -> scalar value/lookahead
```

Training uses the first T-1 observed-history slots as the prefix and the last
real frame as the query. Episode starts inherit first-frame replication padding.
In inference, the query is an analytic candidate successor. It
never changes the persistent observation history. All 80 queries share one
prefix encoding. The full-window control updates a disposable state copy with
the query; it also never persists hypothetical observations.

KDA evidence fusion compares `f + gate(f,m,e)*m` with the same-capacity generic
gate using zero evidence. GDN2 evidence revision supplies `e` to the existing
channel-wise erase/write projections, compared with zero evidence at exactly
the same parameter count. Here `e` is causal observed velocity innovation,
signed speed change and a validity bit, computed only from real prefix frames.
It is not a hidden intent, goal change timestamp or future truth.

There is one actor memory, not separate motion/context networks. Channel-wise
gates do not guarantee semantic motion/context separation or safe forgetting;
that is a hypothesis to test, not an architectural property already proved.

The compact cells implement the exact MIT FLA reference recurrence and omit
language-model convolutions, hybrid attention and large decoders. They do not
claim to reproduce the full Kimi Linear or GDN2 language-model architecture.
They need no additional CUDA package. Credit/license: vendor/FLA_LICENSE;
reference commit 9f38d24980c46d46bd38614e743cdacd21906578.

| Arm | Temporal/read interface | Parameters |
| --- | --- | ---: |
| actor_gru | Original actor-first GRU | 300,417 |
| gru_evidence | GRU prefix/read and evidence fusion | 366,593 |
| kda_full | Compact KDA with disposable query write | 268,177 |
| kda_read | KDA read-only query, current residual | 268,177 |
| kda_gate | KDA generic gated residual | 301,585 |
| kda_evidence | KDA evidence-gated residual | 301,585 |
| gdn2_read | GDN2 read, zero evidence at write gates | 335,241 |
| gdn2_revision | GDN2 evidence-conditioned write gates | 335,241 |

The two evidence-specific contrasts are parameter matched; comparisons between
different substrates are not. GRU evidence fusion is the strong cheap control.
Matrix-state capacity is also different: at these dimensions KDA/GDN2 store
40,960 floats versus the original GRU's 1,280, not a matched state-size control.
The frozen protocol uses seeds 191/223, the same immutable 128-episode ORCA
dataset, 50 IL epochs, 1,000 online MC episodes, four updates/episode, width128,
depth2, T24, reward/actions/simulator and 96 development cases/model. These
seeds are a new architecture pilot, not fresh confirmation of earlier trials.
Two seeds and reused cases cannot establish METHOD_ENTRY_FOUND.

```bash
python -m experiments.temporal_memory queue --data data/demonstrations.pt \
  --root outputs/memory_pilot --device cuda
python -m experiments.temporal_memory summarize --root outputs/memory_pilot
python -m experiments.temporal_memory latency --root outputs/memory_pilot \
  --seeds 191 --device cuda
python -m experiments.temporal_memory events --root outputs/memory_pilot \
  --seeds 191 --device cuda
```

Tests compare recurrence and gradients, official reference equations, causal
evidence, masks/re-entry, read-only candidate queries, shared-prefix versus
full-window values/gradients, and native candidate scores. Operator provenance
is not novelty: actor memory, separate current/history consumption and generic
gating have close priors, including ReCAT (https://intuitive-robots.github.io/ReCAT/).
TRACER (https://arxiv.org/html/2609.18776v1) also separates executed evidence
updates from candidate-trajectory queries in social navigation. That principle
is not a novel claim of this implementation.
Navigation results and evidence-specific ablations must justify any narrower
claim before the architecture is selected as a paper method.

The latency replay also measures the original trained actor GRU with a shared
prefix computation, preserving its value function. This prevents attributing
generic prefix reuse to a new memory operator. Natural-event shadow comparisons
use common roots from the first legal near-motion event in each pre-fixed parent
episode, not the best events for a new arm. They remain exploratory supporting
evidence, not a replacement for a negative paired SR result.

## Completed Memory Pilot

All eight arms finished both paired seeds (191/223): 16 final checkpoints,
50 IL epochs and 1,000 online MC episodes each, with 1,536 fixed evaluation
episodes in total. This cohort is separate from the older processing-order
experiments. No reward, action support, demonstration data or training budget
was changed after observing outcomes.

| Arm | SR | Collision | Timeout | Successful time (s) | RTX 4090 score (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| actor_gru | 82.81% | 6.25% | 10.94% | 18.20 | 3.013 |
| gru_evidence | 78.12% | 7.29% | 14.58% | 17.56 | 3.839 |
| kda_full | 74.48% | 15.62% | 9.90% | 19.51 | 11.678 |
| kda_read | 80.21% | 6.77% | 13.02% | 22.56 | 9.590 |
| kda_gate | 84.90% | 7.81% | 7.29% | 20.56 | 9.726 |
| kda_evidence | 75.52% | 6.25% | 18.23% | 23.47 | 9.716 |
| gdn2_read | 77.08% | 8.85% | 14.06% | 20.46 | 9.788 |
| gdn2_revision | 75.00% | 7.29% | 17.71% | 22.70 | 9.799 |

The parameter-matched mechanism tests are negative in both seeds:

- KDA evidence versus generic gate: SR -8.33 / -10.42 pp; mean -9.38 pp,
  timeout +10.94 pp. Adding motion evidence does not justify this gate.
- GDN2 evidence revision versus zero-evidence update: SR -1.04 / -3.13 pp;
  mean -2.08 pp, timeout +3.65 pp.
- Against the original actor GRU, the custom KDA/GDN2 arms lose 7.29 / 7.81 pp
  mean SR. Neither beats the GRU evidence control either.

Generic KDA gating has the highest mean SR, but its gain over actor GRU is only
+2.08 pp with one positive seed and one tie. Successful-episode time rises
about 13%; different success sets make this a descriptive, not causal, time
comparison. Its pooled 20-human SR is 76.56% versus 65.63% for actor GRU, but
this secondary reused-case slice does not rescue the failed primary gate or
establish a social-specific mechanism.

Every frozen contrast returns NO_CONSISTENT_PILOT_GAIN. This is
**VERSION_NEGATIVE, not FAMILY_NEGATIVE**; two seeds cannot establish permanent
dominance or a paper-ready method. No extra fresh training was launched.

### Cost and Validation

The timing table measures the complete 80-action score on an otherwise idle
RTX 4090, PyTorch 2.9.1+cu128, one CPU thread, 20 warmups and 100 synchronized
samples. The output-equivalent cached actor GRU takes 3.373 ms, so generic
prefix reuse is not a GPU speedup in this workload. KDA/GDN2 are roughly three
times slower than the original GRU here. These compact PyTorch cells are not
optimized official FLA kernels; this result does not benchmark those kernels.

On the i7-1165G7 laptop (PyTorch 2.4.1, one thread), actor GRU / cached GRU
take 64.032 / 5.530 ms. KDA evidence / GDN2 revision take 11.403 / 11.276 ms.
Thus the CPU caching benefit is already available without a new operator.
The local RTX 3060 replay is supplemental only: an unrelated RustDesk compute
process was active, so it is not an idle-device performance claim. Timings
across different devices/PyTorch versions are not pooled.

Training wall times per model are 573-778 s for the GRU arms and 1,446-2,063 s
for the matrix-memory arms. Varying concurrent worker counts and episode lengths
make these descriptive resource records, not matched throughput estimates.
Peak allocated memory is 692-724 MiB / 1,889-1,980 MiB respectively.

The common-root shadow covers 12 native parent episodes, 4,365 person-frames
and six first legal near-motion events. Over three-second continuations,
KDA evidence versus generic gate has three progress wins and three losses;
GDN2 revision versus its matched control has zero wins and four losses
(>=0.05 m). All branches are collision-free. Changed root actions therefore
do not establish recovery value or selective motion/context retention.

All 16 artifacts were checked for finite weights/losses, 50 IL epochs,
1,000 RL episodes, identical case sets and the shared data checksum. Source
and result/checkpoint/log hashes were compared with the training host. Normal
CLI loading was also checked for both custom checkpoints, not used as extra
performance evidence. The final local suite passes 50 tests, including legacy
Mamba parity and the NumPy-to-JSON shadow-export regression. The laptop runs
50 tests with 46 passing and four explicit optional-asset skips. All remote
artifacts were retrieved and checksum-verified before this run's temporary
4090 workspace was removed; existing environments were left untouched.

Full checkpoints and records remain local in outputs/memory_pilot, excluded
from Git. The existing strategy report contains the detailed paired contrasts.
The useful delivered result is a tested, compact architecture and reproducible
negative mechanism comparison, not a successful new navigation algorithm.

## Frozen KDA Gate Diagnostic

```bash
python -m experiments.temporal_memory gate-diagnostic \
  --root outputs/memory_pilot --device cuda
```

No new training: 38 common roots from 12 fixed native parent episodes, using
uniform ticks plus six first arrived near-motion events; both trained seeds.
KDA gating changes memory **readout**, not erase/write. The GDN2 update
mechanism is not tested by this read-gate diagnostic.

Removing only explicit motion evidence changes 0/76 candidate selections;
removing motion and validity changes 1/76. The direct mean gate change from
motion is about 0.00063. This does not support attributing the 9.38 pp SR gap
to harmful runtime motion gating on these states. Entire trained models differ,
and online MC refinement collects policy-dependent trajectories.

In the generic model, constant per-channel gates change 7/76 selections, a
uniform 0.5 gate changes 18/76, and no attenuation changes 35/76. Its mean gate
is 0.56, without broad saturation. This suggests readout scale calibration,
not demonstrated semantic stale-motion erasure. Constants use this same root
cohort; interventions are potentially out of distribution, final rankings
include the inherited safety filter, and no closed-loop improvement is claimed.
Raw diagnostics remain in outputs/memory_pilot/gate_diagnostic.json. A new
read-only intervention/restoration regression brings the local suite to 51
passing tests.

## Frozen Static-versus-Dynamic Follow-up

This follow-up trains read-only KDA with coefficient1, 128 learned
state-independent sigmoid channel scales, or the existing generic dynamic
gate. Actor GRU remains an external reference. Static scales initialize at0.5;
all shared KDA weights have identical initialization for a paired seed.
Parameters: 268,177 / 268,305 / 301,585; actor GRU has300,417. Capacity
differences are reported, not hidden using unused new parameters.

The separate frozen protocol uses four new seeds307/331/359/383 and cases
400-415 in circle/square with5/10/20 humans. Data, reward, actions,
50 IL epochs and1,000 MC-RL episodes are unchanged. Diagnostic IL50/RL500
snapshots are retained, but only the final checkpoint is eligible for the
primary comparison. Online trajectories still depend on the learned policy.

```bash
python -m experiments.temporal_memory queue \
  --protocol experiments/temporal_scale_protocol.json \
  --data data/demonstrations.pt --root outputs/scale_followup --device cuda
python -m experiments.temporal_memory summarize \
  --protocol experiments/temporal_scale_protocol.json --root outputs/scale_followup
```

Dynamic versus static is the primary contrast. A meaningful gain is at least
3 pp SR with3/4 positive seed pairs and the unchanged safety/progress limits.
Practical equivalence requires the paired90% t interval inside +/-3 pp for
aggregate SR only; failure to find a gain is not equivalence. The protocol was
frozen before any outcomes were inspected. Ordinary dynamic gating is not
automatically a new social-navigation mechanism.

### Four-seed Results (4 October 2026)

All16 models completed the frozen budget and1,536 evaluations. Only final
checkpoints are compared; neither intermediate snapshots nor the earlier
two-seed pilot are pooled into these results.

| Readout/reference | SR % | CR % | Timeout % | Successful time s | Successful path m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actor GRU | 79.17 | 10.16 | 10.68 | 16.62 | 11.69 |
| KDA read, coefficient1 | 76.82 | 11.72 | 11.46 | 22.21 | 16.12 |
| KDA static channel scale | 79.43 | 9.64 | 10.94 | 20.96 | 14.98 |
| KDA dynamic gate | 79.69 | 10.94 | 9.38 | 19.43 | 14.56 |

The primary dynamic-minus-static SR differences for307/331/359/383 are
-1.04 /0.00 /-7.29 /+9.38 pp. Mean +0.26 pp; paired90% interval
[-7.83,+8.35] pp. Only one positive pair, two negative and one tie:
**NO_CONSISTENT_PILOT_GAIN**, and practical SR equivalence is **not** established.
All five pre-fixed contrasts fail the pilot-gain rule. Dynamic-minus-GRU is
only +0.52 pp with one positive pair and16.90% longer successful time;
static-minus-GRU is +0.26 pp with26.09% longer successful time. Successful
time/path averages concern different surviving episode sets, not paired
progress equivalence. Six-cell supporting results remain in the raw summary.

| Complete80-action score | Idle4090 median ms | Laptop CPU median ms |
| --- | ---: | ---: |
| Actor GRU, original batched implementation | 3.02 | 56.43 |
| Actor GRU, mathematically equivalent prefix reuse | 3.37 | 5.38 |
| KDA read | 9.65 | 11.00 |
| KDA static | 9.65 | 11.07 |
| KDA dynamic | 9.74 | 11.47 |

These are100 repetitions after20 warmups, one pre-fixed five-human root,
T24 and no simulator/smoothing time. Server timing starts after all training
processes exit; CPU timing uses the laptop. KDA's apparent CPU advantage over
the unreused GRU is absorbed by prefix reuse; no efficiency advantage is found
over the stronger compute control. This compact recurrence is not the optimized
FLA kernel. KDA actor state is160 KiB versus5 KiB for GRU at this configuration.

Actual process training time is11.77-15.15 min for GRU,34.80-39.29 for KDA
read,30.06-36.32 for static and28.30-41.77 for dynamic. Concurrent load varies
from six to eight jobs; these are recorded costs, not isolated throughput
benchmarks. Summed overlapping training/evaluation times are7.94/0.79 process
hours, not GPU-hours. Peak allocated memory per training process is724 MiB
for GRU and1,890 MiB for KDA.

All48 checkpoints reload with exact configuration/parameter counts and finite
weights. Each log contains50 IL epochs and1,000 RL episodes; every model has
the same96 expected cases. The learned static coefficients finish near0.501,
with the full four-seed range0.4982-0.5051, so this control is close to uniform
attenuation rather than a strongly differentiated channel calibration.

**Interpretation:** the old two-seed dynamic-gate advantage does not replicate
as a stable gain here. This neither proves static/dynamic equivalence nor
rejects temporal navigation, actor memory or KDA as a family. It does not
support selective motion-evidence revision or a new method claim. Keep GRU
as the health/reference baseline. The next justified diagnosis is to locate
the divergence using retained IL50/RL500 snapshots under the same evaluator,
then test one identified replay/readout-contract issue; do not search hundreds
of outcome-selected gate variants or rescue a favorable seed.

Scientific source is frozen at6dde31e. Local results are in
/home/abc/workspace/shixu/outputs/scale_followup, including the protocol/source
manifest, paired summary, full episode records, learning logs, three checkpoints
per model and GPU/CPU latency arrays. Code is versioned; weights are not added
to Git. All98 remote raw artifacts and nine scientific source files match
local SHA256 checksums; laptop timing also matches its original checksum.
The server-only temporary workspace is removed after verification, with the
installed environment left intact. No additional training or architecture
is started by this analysis.

## Occlusion Research Loop

The occlusion experiments use legal measured/retained tracks, five-person
ORCA IL and online MC refinement, followed by reloaded-final-weight tests on
5/10/20 people in circle and square. The frozen primary endpoint is equally
weighted 10/20-person success, with collision, timeout and progress checks.
Both the ORCA teacher and the student receive legal measured/CV-retained
tracks from the same observation interface; never-seen pedestrians are excluded.
Temporary server outputs are copied locally before deletion. Development
results are not final evidence: a promising mechanism needs unseen seeds and
cases, with all controls retrained on the same host.

The V5 private-successor comparison completed all four seeds419/443/467/491.
Overall SR/CR/timeout are75.00/10.16/14.84% for measured-only KDA branches,
77.34/10.68/11.98% for CV-pseudowrite branches, and85.94/5.99/8.07% for the
strongest completed context-GRU reference. Their primary SRs are71.48/74.61/
85.55%. Measured-only branching loses to that reference in4/4 pairs. Neither
private branching nor excluding legal CV writes establishes a navigation gain.
Results remain in outputs/occlusion_v5; this is a version-level negative result.

A separate frozen-consumer shadow replaces only currently retained hidden
positions/velocities with current simulator truth, without introducing unseen
people or modifying tracker memory. Four-seed primary gain is only0.39 pp
(one positive, two negative, one tie). This is not a full-future upper bound:
it shows no large demonstrated hidden-state accuracy headroom for that frozen
consumer, not that temporal reasoning or occlusion handling is unnecessary.
Records remain in outputs/occlusion_v1/*/current/truth_retained.json.

V6 tests physical actor memory before current candidate geometry/goal fusion.
It uses the official-shaped KDA no-short-convolution mixer, verified against
the pinned FLA layer, rather than claiming that mixer or its output gate as new.
All motion-family models receive the same physical features and legal elapsed
interval input. Only the custom elapsed-clock arm scales channel log-decay by
the real interval; vanilla KDA uses unit decay per measurement. Controls include
same-placement GRU, CV pseudowrites, identical-capacity zero motion history,
current-only and the strong completed context-GRU. There is no extra loss or
prediction model. The clock mechanism is a hypothesis, not a demonstrated win.

```bash
python -m experiments.occlusion queue \
  --protocol experiments/occlusion_motion_protocol.json \
  --root outputs/occlusion_v6 --data outputs/occlusion_v6/demonstrations.pt \
  --arms motion_gru motion_kda motion_elapsed motion_imputed motion_nohistory \
  --device cuda --workers 2
```

Generic time-aware recurrence already exists in GRU-D and time-aware LSTM;
actor memory and delta-rule erase/write are also existing mechanisms. The
remaining question is whether legal observation-time semantics and this
physical/current fusion improve closed-loop navigation beyond those controls.
No novelty or safety guarantee is earned by passing numerical tests.

V6 scientific source is frozen at54d19cc. The 4090 runs PyTorch2.9.1/cu128,
and the3060 runs2.1.0/cu121; development comparisons retain this host boundary.
The CUDA packed-GRU test on2.9.1 differed from explicit stepping by2.36e-4
with default cuDNN TF32, and3.86e-6 with TF32 disabled. Float64 validates the
recurrence separately; production float32 kernels are not bitwise identical.
The archived optimized seed419 GRU replay changes controls in9/96 episodes
but changes no terminal outcomes. These checks cannot justify universal action
parity. Fresh confirmation must use one frozen implementation on one host.

### Physical-memory Results and the Next Readout Test

All20 V6 models completed50 IL epochs,1,000 online MC episodes and96
reloaded-final-weight evaluations each. Source, data, finite weights and the
complete episode keys are checked; all77 server artifacts match local checksums
before the temporary V6 workspace is removed.

| Model | Overall SR % | CR % | Timeout % | Primary10/20 SR % |
| --- | ---: | ---: | ---: | ---: |
| Current legal CV-track reference | 81.77 | 4.95 | 13.28 | 79.30 |
| Context-GRU reference | 85.94 | 5.99 | 8.07 | 85.55 |
| Physical-memory GRU | 87.50 | 4.43 | 8.07 | 83.59 |
| Physical KDA, unit observation clock | 80.99 | 3.65 | 15.36 | 76.56 |
| Physical KDA, elapsed clock | 80.99 | 2.86 | 16.15 | 76.95 |
| Physical KDA, CV pseudo-writes | 83.07 | 5.47 | 11.46 | 78.13 |
| KDA capacity control, zero committed history | 86.72 | 6.51 | 6.77 | 83.59 |

Elapsed-minus-unit primary differences are-9.38/+4.69/+7.81/-1.56 pp:
mean+0.39 pp, two positive and two negative. Elapsed-minus-physical-GRU is
-6.64 pp, with two losses and two ties. Fewer collisions are accompanied by
more timeouts, so this version does not meet the fixed success/progress rule.
The zero-history arm still receives legal interval/age input. These are
development results, not a family rejection or evidence that history is useless.

On836 uniformly sampled demonstration windows per checkpoint, first-layer
cross-actor key cosine averages0.297-0.327 across the four trained unit-clock
KDA models. Thus the earlier near-identical-address symptom is no longer
observed here; this statistic does not certify useful retrieval. Removing all
committed memory changes scalar values substantially, including hidden-track
windows, but value sensitivity does not establish better action ranking.

A separate frozen action diagnostic replays12 pre-fixed current-reference
episodes per seed and samples every eighth control step. Across626 states,
338 contain retained hidden actors. Zeroing only those actors' committed memory
changes79/338 actions (23.37%); zeroing all committed memory changes344/626
(54.95%). The value shift is mostly common across candidates, but the smaller
action-dependent component often changes the winner. This rules out a purely
common-offset explanation; it does not show that the changed actions are better.
The query geometry, age inputs, reward, action support and trained weights stay
fixed. All replayed terminal outcomes match the archived current-reference
episodes. Raw records are in outputs/occlusion_v6/action_content_*.json.

The corresponding frozen closed-loop shadow uses those first two cases per
cell (12 episodes per seed,48 total), without selecting strong interactions.
Original SR/CR/timeout are83.33/6.25/10.42%; masking only hidden-actor memory
gives79.17/2.08/18.75%, and masking all memory gives54.17/6.25/39.58%.
Hidden-memory removal reduces SR in two seeds, improves it in one and ties in
one. Memory matters to this trained model, with a safety/progress trade-off;
this intervention can be out of distribution and does not establish superiority
over a separately trained no-history model. Raw continuations and matched parent
records remain in outputs/occlusion_v6/closed_content_*_kda.json.

A separate four-seed CPU shadow checks the fixed2-second retention deadline.
Only previously seen expired actors still inside the legal history horizon
are eligible: current truth adds0.39 pp primary SR, while extending ordinary
CV retention adds0.78 pp. Of602 expired person-frames with a last measurement
still inside the legal prefix,31 are within2 m of the robot. This demonstrates
a lifecycle limitation but no large frozen-consumer headroom or residual beyond
the simple CV control. It does not justify another expiration architecture.
Raw records are in outputs/occlusion_v6/expiry_shadow.

V7 therefore replaces the readout interface, not another gate. Measured physical
actor streams write the same KDA memory; existing current robot/human fusion
moves before access and creates a different query for each candidate action.
The read-only arm contracts that query with the pre-existing matrix. It never
treats a hypothetical successor as a new measurement. A private-update KDA
control uses exactly the same fusion placement and parameters; a same-placement
GRU and a zero-committed-history control are also trained. No extra network,
prediction target, reward or data is introduced. Physical V6 references preserve
their original readout and budgets.

The mechanism hypothesis is that action-relevant retrieval can improve the
use of physical history beyond a candidate-independent actor summary. It is
not presumed true. The proposed read-before-write primitive is already present
in [DRAM](https://arxiv.org/abs/2609.32453), and task-conditioned retrieval is
not new. [Advantage-Driven Explicit Memory](https://arxiv.org/abs/2608.25610)
retrieves recurrent navigation experiences across episodes, unlike this bounded
per-actor measured stream. [Kimi Linear](https://arxiv.org/abs/2510.26692)
provides the mixer. A possible paper claim must concern the candidate-query/
occluded-actor interface and validated navigation benefit, not qS, KDA, actor
identity or generic read/write separation alone. Novelty remains unearned.

V7 is frozen at36d1702 with the same128 demonstrations, four paired seeds,
5-person training and5/10/20-person evaluation. Its pre-outcome protocol is
experiments/occlusion_query_protocol.json; results are in outputs/occlusion_v7.
All controls must remain in the comparison. Fresh seeds/cases remain untouched
until an eligible development winner is chosen.

These budgets complete the IL-to-online-MC pipeline, not the original camrl
training schedule:50 IL epochs use128 successful demonstrations, followed by
1,000 online episodes. At the final episode epsilon is still approximately
0.1335 on its1,500-episode decay schedule. They do not establish convergence.
If readout placement fails, a justified next test is one pre-fixed longer RL
budget for all competing arms, with the same demonstrations and architecture;
changing data volume and training budget together would not isolate the cause.

The pre-outcome budget protocol is experiments/occlusion_budget_protocol.json.
It fixes3,000 MC episodes for physical KDA, contextual GRU, same-placement GRU
and the original zero-committed-history KDA capacity control, keeping128 IL
demonstrations and50 IL epochs. No architecture changes are bundled with this
test. Matched final-IL checkpoints and all50 original IL log entries are reused
only after complete seed/configuration/hash checks; online replay is rebuilt
from the same demonstrations and all3,000 MC episodes are run, not resumed from
an incomplete RL buffer. The4090 handles419/443/491 and the3060 handles467.
Neither this budget nor the old1,000-episode budget is described as converged.
Fresh confirmation remains reserved.

### Candidate-query Results

All16 V7 models completed their fixed budgets and96 held-out episodes each.
All62 remote artifacts match local SHA256 checksums; final weights, complete
50-epoch/1,000-episode logs, shared data and episode keys are verified.

| Query interface | Overall SR % | CR % | Timeout % | Primary10/20 SR % |
| --- | ---: | ---: | ---: | ---: |
| Same-placement candidate-query GRU | 87.24 | 5.47 | 7.29 | 82.81 |
| KDA private candidate update | 71.61 | 8.07 | 20.31 | 70.31 |
| KDA read-only candidate query | 75.78 | 9.90 | 14.32 | 74.61 |
| Read-only query, zero committed history | 81.25 | 5.99 | 12.76 | 77.34 |

Read-only-minus-same-placement-GRU primary differences are
-26.56/-4.69/0.00/-1.56 pp: three losses and one tie. Against contextual GRU,
the mean is-10.94 pp, with4.69 pp more collisions and6.25 pp more timeouts.
Read-only access beats the private KDA update on aggregate SR, but not the
pre-fixed safety/seed-consistency rules or the stronger controls. This version
does not establish better retrieval, navigation or a new method. Its result is
VERSION_REQUIRES_DIAGNOSIS_NOT_DIRECTION_REJECTED; results and checks remain in
outputs/occlusion_v7. V8 isolates the learning-budget question rather than
adding another query gate or selecting a favorable V7 seed.

### Query-contract Check During the Budget Run

An independent held-out collection contains64 successful legal-observation
ORCA episodes from cases13000 onward. Final-IL snapshots, not final-RL weights,
are compared against expert continuation returns on762 uniformly sampled
transitions;48 active-support changes are excluded before model evaluation.
Each transition has the same legal prefix and target. Only the final query
changes from the actual next observation to the planner's CV successor.

| Four-seed mean MSE | Observed next | CV next | CV with fresh-age shadow |
| --- | ---: | ---: | ---: |
| Context GRU | 0.00230110 | 0.00231515 | 0.00232848 |
| Physical GRU | 0.00219292 | 0.00219781 | 0.00220714 |
| Physical KDA | 0.00240531 | 0.00241053 | 0.00241885 |

The age shadow resets only the candidate age of actors measured at the root;
it is fictitious and is not a proposed deployment fix. These results do not
show a substantial aggregate query-mismatch penalty in this expert cohort.
They therefore do not justify treating that mismatch as the demonstrated cause
of poor navigation, or changing the training target on that basis. This check
does not cover on-policy RL errors,20-person observations or action quality.
The helper is experiments/query_contract_probe.py; raw predictions, data hashes
and checkpoint hashes remain in outputs/query_contract.

V8 used one frozen source for every arm. The resumed seed467 physical-GRU
prefix exactly matches its earlier terminal outcomes and losses over706 checked
episodes. The older context-GRU run used a different but semantically equivalent
packed-GRU implementation, so its resumed prefix is not an exact replay. Any
old-versus-new context-GRU difference cannot be attributed exclusively to budget;
the current V8 within-run comparison remains matched. Fresh confirmation, if
earned, must train all arms from scratch on one frozen implementation and host.

### Budget Results and Round Closure

All16 V8 models completed50 IL epochs and3,000 online MC episodes, then96
final-weight tests each. All95 remote artifacts match local SHA256 checksums;
the server RAM workspace is removed. Complete records are in outputs/occlusion_v8.

| Model | Overall SR % | CR % | Timeout % | 5-person SR % | 10-person SR % | 20-person SR % | Primary10/20 SR % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Context GRU | 85.42 | 6.51 | 8.07 | 93.75 | 84.38 | 78.12 | 81.25 |
| Same-placement physical GRU | 86.98 | 6.25 | 6.77 | 98.44 | 86.72 | 75.78 | 81.25 |
| Physical KDA | 86.98 | 5.99 | 7.03 | 98.44 | 87.50 | 75.00 | 81.25 |
| Zero-committed-history KDA | 87.76 | 7.55 | 4.69 | 94.53 | 92.19 | 76.56 | 84.38 |

KDA-minus-context primary differences are-9.38/-7.81/-6.25/+23.44 pp;
KDA-minus-physical-GRU differences are-4.69/-6.25/+6.25/+4.69 pp. Equal
pooled rates are not an equivalence result. Against zero committed history,
KDA loses3.125 pp primary SR and incurs5.08 pp more timeouts while reducing
collisions1.95 pp. No contrast meets the pre-fixed admission rules.

The KDA1,000-episode prefix exactly reproduces V6 for all four seeds. Its
primary SR rises4.69 pp with the larger budget, but only2/4 paired seeds rise;
overall successes increase311 to334, collisions14 to23, timeouts59 to27.
This supports a partial budget effect, not convergence or a stable advantage.

No fresh confirmation or V9/V10 is started. The current version is stopped,
not the entire temporal-model family rejected. The Chinese interim report
OCCLUSION_KDA_INTERIM_REPORT_20261004.md records the final results, evidence
boundaries and diagnostic gaps. No associative-memory necessity, learned
selective revision, or MC-objective bottleneck is established by these results.

### Read-only Closure Audit

The user's five diagnostic questions are answered in section14 of the Chinese
report. No new training, controller or reward change is part of this audit.

```bash
PYTHONPATH=vendor:. python -m experiments.ranking_audit \
  --root outputs/occlusion_v8 --old-root outputs/occlusion_v6 \
  --expert-data outputs/query_contract/demonstrations64.pt \
  --output outputs/query_contract/closure_ranking_audit.json
```

Sixteen failure trajectories were selected by a fixed first-key rule; eight
timeouts and six sufficiently long collisions supply55 shared frozen states.
Both physical models and their IL50/RL1000/RL3000 snapshots receive identical
legal histories and80 candidates. Every source final-weight executed command
matches its archive. The independent true-human two-second geometry reference
holds each candidate velocity constant; it is not an optimal Q or a closed-loop
navigation improvement. Different-time outcomes are not pooled as one benchmark.

Of40 timeout anchors,32 admit a one-step CV-safe-progress action. Ranking by
scalar V selects one in15 GRU versus3 KDA states; native reward-plus-value
selects20 versus8, filtering27 versus15, smoothing20 versus13. This locates
part of the difference before filtering, not specifically inside memory read.
Under a stricter true two-second clearance/progress criterion, native ranking
selects5/20 for GRU versus8/20 for KDA: the contrary evidence rules out calling
KDA's short-term ranking uniformly worse. Pre-collision5–10-second coverage is
insufficient to establish a gradually forming bias.

Final-IL held-out expert-return MSE is0.00219292 for GRU and0.00240531 for KDA.
Post-RL discrepancies against those expert returns are not on-policy value
errors; no archived common online replay or true80-action value reference
supports declaring the MC objective the cause. Earlier RL500 snapshots exist
only for a different full-observation cohort and cannot fill this curve.
The current V8 query uses common physical human successors; candidate robot
actions enter post-read fusion, not KDA q. Learned retrieval semantics and
associative-memory necessity remain unproved. Current tests:91,88 pass and
three optional legacy-asset skips. Main frozen scientific sources are unchanged.


<!-- END PRESERVED SOURCE -->


---

<a id="stage-3"></a>

## Source: outputs/forecast_control_remote_backup_20261004/README.md (historical README section 1)

Original full-source SHA-256: d5ac9d5a389c9f6a37c0f9ddc216d8995770f37cc60863cfa476233835b54058

<!-- BEGIN PRESERVED SOURCE -->
# shixu

A small temporal crowd-navigation research framework extracted from the user's
local camrl Mamba-VL project. It contains the cleaned parent and explicit
temporal architecture experiments, not an established new-method claim.

## One Main Path

```text
legal observations -> frame encoder -> temporal encoder -> scalar value
                    -> original candidate successor/value lookahead -> action

ORCA demonstrations -> Monte Carlo value initialization -> online MC refinement
```

GRU is the default temporal baseline. Mamba is optional and retained only for
legacy comparison. There are no Double-Q/PPO/SAC branches, auxiliary prediction
heads, Bayesian modules, teacher networks at deployment, or fallback backbones.

The explicit actor-first alternative moves temporal encoding before crowd pooling:

```text
identity-bound human histories -> shared temporal encoder -> crowd pooling
                              -> scalar value -> unchanged lookahead
```

The processing-order prototype now compares scene-first and actor-first GRU
using identical aligned observations and exactly the same parameters. Every
human uses the same GRU weights, with independent histories. That trial adds no
dual-memory system, contradiction detector or additional prediction head.
That processing-order experiment remains a baseline, not a selective-revision method.

The observation-only contract consumes robot raw state9 and human observed
motion9 plus presence. It removes redundant legacy relation features rather
than adding a predictor or another memory. The explicit legacy feature contract
remains only for reproducing the first trial and loading its checkpoints.

## Layout

| File | Responsibility |
| --- | --- |
| shixu/features.py | Legacy observation contract and history windows |
| shixu/observations.py | Episode-local observed association keys, never numeric ID features |
| shixu/model.py | Frame encoder, replaceable temporal encoder, value head |
| shixu/temporal.py | Compact GRU/KDA/GDN2 actor memories; real-write/candidate-read interface |
| shixu/policy.py | Original action support and successor-value evaluation |
| shixu/replay.py | Episode-safe windows and MC targets; no duplicated window archive |
| shixu/runner.py | One runner for collection, training and evaluation |
| shixu/training.py | ORCA value initialization and online MC refinement |
| shixu/cli.py | Explicit collection/evaluation/training commands |
| vendor/crowd_sim | Frozen local simulator dependency |
| experiments/ | Frozen protocol, immutable ORCA collection and paired processing-order trial |

## Occlusion Development Loop

The new `occlusion` architecture removes the five-person input cap. It uses
episode-local identity slots, actual-measurement write masks and a separate
retained-track read mask. A previously observed actor can remain relevant while
occluded; an actor that has never been seen cannot enter the model. Missing
positions use the last legally measured velocity for at most two seconds.
Predicted positions and candidate successors are read-only queries, not new
measurements. There is one shared actor memory and no external memory gate,
auxiliary predictor, dual-memory branch or changed reward.

```text
body-occluded observations -> legal track histories -> shared KDA memory
                          -> retained-track candidate reads -> scalar value
                          -> inherited 80-action value lookahead
```

This is a functional research prototype, not an established novelty or
performance claim. The common retention interface is also used by the
current-state and recurrent comparisons. Previously reported full-observation
results are not occlusion results.

```bash
# Pin the vendored simulator when another CrowdNav is installed locally.
export PYTHONPATH=vendor:.
python -m unittest discover -s tests -v
python -m experiments.occlusion collect \
  --data outputs/occlusion_v1/demonstrations.pt
python -m experiments.occlusion queue \
  --root outputs/occlusion_v1 --data outputs/occlusion_v1/demonstrations.pt \
  --device cuda --workers 4
python -m experiments.occlusion summarize --root outputs/occlusion_v1
```

`experiments/occlusion_protocol.json` freezes shared demonstrations, four
paired development seeds, 50 IL epochs, 1,000 online MC-RL episodes and
5/10/20-person evaluation. Every reported model is reconstructed and loaded
from its final saved checkpoint before evaluation. Development cases guide
diagnosis; fresh seeds and separate confirmation cases remain reserved until
an architecture is selected. A negative version is diagnosed, not relabeled
as a failed research family. Raw weights, logs and episodes remain local.

The second frozen version, `occlusion_observation_protocol.json`, changes only
the read clock. Memory is read from the latest legal history frame once per
actor, then shared by all candidate actions. Candidate geometry remains in the
spatial value encoder, but it no longer changes the actor's temporal read vector.
The weights and parameter count are unchanged by this switch. Both versions
receive byte-identical episode observations and rewards. The current-track
reference is reused because it has no learned temporal read; the temporal
arms are retrained through the complete IL/MC-RL schedule.

```bash
python -m experiments.occlusion collect \
  --protocol experiments/occlusion_observation_protocol.json \
  --data outputs/occlusion_v2/demonstrations.pt
python -m experiments.occlusion queue \
  --protocol experiments/occlusion_observation_protocol.json \
  --root outputs/occlusion_v2 --data outputs/occlusion_v2/demonstrations.pt \
  --arms gru kda --device cuda --workers 4
python -m experiments.occlusion summarize \
  --protocol experiments/occlusion_observation_protocol.json \
  --root outputs/occlusion_v2
```

The observation-clock design is a hypothesis under test, not a correctness
repair. Both versions leave the stored matrix state unchanged during candidate
evaluation. Action-conditioned reads can legitimately retrieve different
information for different decisions even when humans do not react to the
robot. The new bias separates a shared temporal read vector from
candidate-conditioned geometry evaluation and avoids replicating each actor's
matrix memory 80 times. Frozen-weight read ablations are diagnostic, not
substitutes for retraining or proof of better navigation.

The first full four-seed occlusion trial is complete (384 held-out development
episodes per arm, not the earlier full-observation trial):

| Model | Overall SR | CR | Timeout | 10/20-person SR |
|---|---:|---:|---:|---:|
| CV-track current-value | 81.77% | 4.95% | 13.28% | 79.30% |
| Actor-GRU | 77.34% | 6.25% | 16.41% | 75.78% |
| Actor-KDA, candidate read | 78.39% | 11.20% | 10.42% | 75.78% |

KDA does not pass: primary SR is unchanged against GRU, while primary collision
increases by 5.86 percentage points. A frozen-weight switch to observation-clock
reads also worsens seed 443 overall SR from 70.83% to 56.25%; this is a
distribution-shifting diagnostic, not a trained comparison. The second version
must therefore earn its own result through matched IL and RL.

Completed final-IL checkpoints may be moved to a faster host using `--il-root`.
The experiment verifies seed/configuration and all IL log epochs, restores the
replay sampling stream, then starts a fresh, full-budget RL run. Interrupted RL
work is archived and charged separately; it is not used for checkpoint selection.
An exact CPU pipeline test checks identical full-run versus final-IL-reuse RL
updates. Different GPU/software environments can still introduce numerical
differences and are recorded with each run.

The complete observation-read trial is negative: overall SR is 63.54% for
KDA versus 79.43% for GRU; primary 10/20-person SR is 64.45% versus 76.56%.
KDA loses primary SR in all four paired seeds. Fewer collisions come with
substantially more timeouts (30.47% overall), so this version is not retained
as the candidate architecture. Shared observation reads are not an established
correction to the original action-dependent read design.

The next structural test, `occlusion_context_protocol.json`, returns to the
original candidate clock and moves the existing attention before actor memory:

```text
real actor observations -> attention (actor outputs, no scene pooling)
                        -> shared, measurement-masked KDA writes
candidate geometry     -> same attention -> read each actor memory
                        -> masked max -> scalar value -> original lookahead
```

There is still one attention and one temporal operator. The parameter counts
are unchanged. Prefix attention keys use actual measurements; hidden tracks
cannot supply fresh write evidence. Query attention can use legally retained
CV tracks. The non-temporal current-value arm is mathematically unchanged;
its existing saved weights reproduce every executed action and terminal outcome
in all 96 seed-419 development episodes after the helper refactor.

This tests whether historical neighbour context is useful. The old actor
memory is exactly insensitive to another actor's past when its own history is
fixed. The replacement removes that insensitivity in a fixture, but this is
not proof of navigation headroom or novelty. A separate episode-disjoint
linear probe of 887 natural re-entries does not improve average velocity
prediction over CV, so richer context is not presumed useful in every task.
Archived collision replays also show visible colliders and no margin-safe
action at the final step; they do not establish that occlusion caused the
earlier poor decisions. DS-RNN and [PaS](https://github.com/yejimun/PaS_CrowdNav)
already study temporal/social inference. Moving attention is an experimental
representation choice, not a standalone new-method claim.

The complete context-write trial is negative for KDA: overall SR/CR/timeout
are 62.76/12.24/25.00%, versus 85.94/5.99/8.07% for GRU. Primary 10/20-person
SR is 60.16% versus 85.55%, with KDA losses in all four paired seeds.
On 836 uniformly sampled demonstration windows, seed-443 final KDA keys have
mean inter-actor cosine 0.99926 in this version versus 0.77600 in V1.
Separate per-actor matrices do not prevent homogenization when their inputs
are almost identical. High cosine is a representation diagnostic, not a proof
that it causes all observed navigation losses.

The next test, `occlusion_address_protocol.json`, preserves local features:

```text
actor feature u_i -> u_i + attention(u, measured actors) -> memory content
actor feature u_i -------------------------------------> KDA q/k (custom)
candidate feature + same attention -> retained-memory read -> max -> value
```

All four arms share the residual spatial path. Current-value, contextual GRU,
ordinary residual KDA and actor-addressed KDA are retrained; the old current
reference is not reused because its query representation changes too. The
two KDA arms have exactly the same parameters and initialization. Only q/k
source differs; values, decay and write strength consume contextual content.
At deeper layers both streams receive the same recurrent output. There is
one attention, one shared actor memory and no additional loss or output head.
Existing read/write versions remain available for their archived checkpoints.

This is a mechanism test, not an established new method. Separate-source
delta memory already appears in [DRAM](https://arxiv.org/abs/2609.32453), and
retrieval design is studied in
[Advantage-Driven Explicit Memory](https://arxiv.org/abs/2608.25610).
The hypothesis here is narrower: retaining actor-local addresses while
remembering observed social context helps legal hidden-track action evaluation.
That claim still needs navigation gains and an occlusion-specific analysis.

```bash
python -m experiments.occlusion queue \
  --protocol experiments/occlusion_address_protocol.json \
  --root outputs/occlusion_v4 --data outputs/occlusion_v4/demonstrations.pt \
  --device cuda --workers 4
python -m experiments.occlusion summarize \
  --protocol experiments/occlusion_address_protocol.json \
  --root outputs/occlusion_v4
```

Demonstrations are the same saved observations/rewards as V1, with protocol
metadata retargeted for the matched run; there is no extra training data.
Final IL and RL weights, all failed versions, raw episode controls and consumed
compute are retained. Fresh seeds and unseen confirmation cases are not used
to develop this version.

The complete V4 result is negative. Overall SR/CR/timeout are 68.75/11.72/19.53%
for actor-addressed KDA, 72.40/10.94/16.67% for vanilla residual KDA and
77.08/5.47/17.45% for residual GRU. Primary SR falls by 11.72 pp against GRU;
actor-local addresses do not rescue this version. The final seed-443 address
probe does recover distinct keys (mean inter-actor cosine 0.8144), so loss of
actor addresses alone is not an adequate explanation of the navigation failure.

`occlusion_branch_protocol.json` tests the inherited successor-value interface,
not a new gate or a claim that private branches are novel:

```text
actual measured actor frames -> shared KDA -> real-history matrix
candidate CV successor      -> private one-step KDA calculation
                            -> existing attention/max/value -> original lookahead
```

Candidate computations never commit into real history. The single-step read
is algebraically identical to an explicit private matrix update, but avoids
80 replicated matrices. Its paired control additionally accepts legally
propagated CV tracks as history pseudomeasurements; it receives no hidden truth.
Both new arms retain identical parameters, initialization and IL/RL budgets.
The unchanged V1 current-value and read-only KDA references and the stronger V3
context-GRU reference remain controls, with configuration and action-parity
checks before reuse. Any positive development result still requires new seeds
and unseen cases, with all confirmation arms trained anew.

Engineering optimizations share matrix reads across candidates and pack measured
GRU frames without changing their recurrence. An exclusive laptop CPU read-only
benchmark falls from 5.91 to 2.37 ms for five actors and 27.67 to 9.90 ms for
twenty actors; these are not full-controller latency or navigation gains.
Frozen checkpoint actions and gradient/recurrence tests validate the changes.

## Installation

Install a PyTorch build appropriate for your machine first. Then:

```bash
git clone https://github.com/jinglongjiang/shixu.git
cd shixu
python -m pip install -e .
python -m pip install 'git+https://github.com/sybrenstuvel/Python-RVO2.git'
```

Python-RVO2 needs its normal native build prerequisites. The GRU path does not
require Mamba, Transformers or custom CUDA kernels. Legacy Mamba checkpoints
require the optional mamba-ssm 1.2.0 dependency and a matching CUDA/PyTorch wheel;
do not silently substitute another network if it fails to import.

## Commands

```bash
# Interface checks only; this is not a trained policy result.
python -m shixu.cli smoke --output outputs/smoke.json
python -m unittest discover -s tests -v

# Collect legal, identity-tagged observations and ORCA returns without training.
python -m shixu.cli collect --cases 0 1 --output data/orca.json

# Trained-checkpoint evaluation. Weights are intentionally not uploaded.
python -m shixu.cli evaluate --backbone mamba --device cuda \
  --weights /path/to/rl_model_ep10000_T24.pth --cases 0 1

# Explicitly opt into training.
python -m shixu.cli train --il-episodes 5 --rl-episodes 10 \
  --device cuda --output weights/gru.pt
```

Models must use the same config when comparing them. The optional local-source
regression tests use environment variables CAMRL_PARENT and CAMRL_CHECKPOINT;
they check features, value outputs, actions and history against the original
source. They skip explicitly when those local assets are unavailable.
New checkpoints include their model/observation configuration; evaluation uses
it automatically unless an explicit --config override is supplied.

The matched trial is driven by experiments/temporal_protocol.json, not test
results: four paired seeds, a shared 128-episode successful ORCA dataset,
50 IL epochs, 1,000 MC-RL episodes per arm, and fixed circle/square cases at
5/10/20 humans. The initial legacy contract has 302,337 parameters per arm;
the observation-only contract has 300,417 at the same width 128 and depth 2.
Only the final-budget checkpoint is evaluated. Processing-order prototype
results cannot be represented as a new algorithm or proof of selective memory.

```bash
python experiments/temporal_collect.py --output data/demonstrations.pt
python experiments/temporal_order.py run --seed 17 --order pair \
  --data data/demonstrations.pt --root outputs/temporal_v1 --device cuda
python experiments/temporal_order.py summarize --root outputs/temporal_v1

# One shared interface rescue: same data/budget, subtract legacy derived inputs.
python experiments/temporal_order.py run --seed 17 --order pair \
  --feature-contract observed --data data/demonstrations.pt \
  --root outputs/temporal_v2 --device cuda
```

Run the other seeds in the protocol before requesting the paired summary.
Native experiments assume perfect observed association and retain the original
five-human neural input cap even when the simulator contains 10/20 humans.
Missing observation masks preserve actor state; association errors and
real-world re-identification are not solved by this interface.

## Initial Matched Result

Four paired seeds completed 50 IL epochs + 1,000 online MC-RL episodes per arm,
followed by 96 fixed native evaluations each (768 total).

| Model | SR | Collision | Timeout | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Scene-first GRU | 75.52% | 9.90% | 14.58% | 302,337 |
| Actor-first GRU | 78.39% | 8.33% | 13.28% | 302,337 |

The +2.86 pp mean SR change has only 2/4 positive seed pairs and does not meet
the frozen +3 pp / 3-of-4 direction gate: NO_STABLE_GAIN. Pooled square gains
and smaller actor seed dispersion are exploratory, not a new-method claim.
Same-device RTX 3060 scoring medians are 3.110 / 4.418 ms for scene / actor;
actor-first is not a computation-saving result. No GDN/KDA/revision cell is
claimed successful on the strength of these mixed outcomes.

```bash
python -m experiments.temporal_latency --root outputs/temporal_v1 --seed 17
python -m experiments.temporal_revision_shadow --root outputs/temporal_v1 --seed 17
```

The shadow uses arrived motion evidence and native scene replay. A masked-prefix
intervention is an offline diagnostic, not a trained or deployable revision
policy. Full results/checkpoints stay local under outputs; weights and data are
not committed. The unchanged fresh follow-up used seeds 103/137: SR changes
were +7.29 / -12.50 pp, so the initial seed-dispersion signal did not replicate.
The common observation-only rescue is frozen separately in
experiments/temporal_rescue_protocol.json; its results must not be pooled with
the legacy-contract cohort.

## Completed Observation-Only Rescue

The one permitted rescue subtracts the inherited redundant/incorrect derived
inputs for **both** arms, without changing data, reward, budget or network size.
Four paired seeds again completed 50 IL epochs, 1,000 MC-RL episodes and 96
fixed evaluations per arm (768 evaluations).

| Model | SR | Collision | Timeout | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Scene-first GRU | 75.52% | 11.20% | 13.28% | 300,417 |
| Actor-first GRU | 75.00% | 6.25% | 18.75% | 300,417 |

SR differences are +3.13, +5.21, -5.21 and -5.21 pp across seeds
17/29/43/71. The mean is -0.52 pp with 2/4 positive pairs:
**NO_STABLE_GAIN** under the unchanged gate. Collision decreases in all four
pairs, but timeout increases; this is a safety-progress operating-point signal,
not proof of better navigation. Pooled 20-human gains also remain only 2/4
seed-positive. Same-device scoring medians are 3.012 / 4.163 ms (scene / actor),
so actor-first is about 38% more expensive in this workload.

The legal observed-change shadow finds four first events in 12 native
episodes: targeted actor-history truncation changes no root rankings and gives
no safe progress gain >=0.05 m. Selective-revision headroom remains unproven;
the small masked-prefix intervention does not reject the research family.
Attention and pooling both move relative to recurrence, so this comparison
does not isolate identity continuity alone.

Reserved fresh rescue seeds 191/223 were not run within that study because the
primary gate failed. That study added no GDN/KDA, new reward or extra teacher.
Across the separate initial, fresh and rescue cohorts, 20 models and 1,920
matched evaluation episodes are retained locally. None is relabeled as a new
method. All 38 local tests pass with the original comparison assets configured;
the laptop passes 35 tests with three explicit original-asset skips.

```bash
python experiments/temporal_order.py summarize --root outputs/temporal_v2
python -m experiments.temporal_latency --root outputs/temporal_v2 --seed 17
python -m experiments.temporal_revision_shadow --root outputs/temporal_v2 --seed 17
python -m shixu.cli evaluate --weights outputs/temporal_v2/17/actor/model.pt \
  --device cuda --cases 0 1
```

## Baseline Boundary

The source baseline comes from the user's
CrowdNav(20260511_last_version_mamba_vl).zip, not the later Bayesian-replaced
active camrl directory. The simulator preserves that archive's behavior;
only trailing whitespace is cleaned.
Its CrowdNav foundation is attributed in vendor/CROWDNAV_LICENSE.

The inherited deterministic baseline uses 80 moving actions, dt=0.25 s,
24-frame history, and r+0.99V lookahead. The archive's evaluation settings also
include clearance filtering, a risk penalty and action smoothing. They are
retained explicitly in shixu/default.ini; this is not a reproduction of paper
statistics based on a few episodes.

Legacy metadata indices and spatial relational-coordinate conventions are
preserved for checkpoint parity. Their audit is separate from method novelty;
changing them together with a new memory would confound that comparison.

The new training runner preserves the IL-to-MC-value-learning formulation, not
every historical launcher's behavior: observations are recorded even during
exploratory controls, teacher state is cleared between episodes, and test-case
scheduling is explicit. All new training arms must share this runner. Legacy
training numbers cannot be attributed to this cleanup without matched reruns.

Simulator IDs are association keys attached to observed states, not neural
features. Human goals/future states are not written into deployable inputs.
Weights, data, videos, credentials and old experiment artifacts are excluded
from version control.

## Explicit Memory Architecture Pilot

A separately authorized pilot compares two mechanisms without assuming the
newer operator is better:

```text
observed actor prefix -> one shared GRU/KDA/GDN2 -> per-actor state
candidate successor  -> query that state       -> current feature + memory
                     -> original attention/max pool -> scalar value/lookahead
```

Training uses the first T-1 observed-history slots as the prefix and the last
real frame as the query. Episode starts inherit first-frame replication padding.
In inference, the query is an analytic candidate successor. It
never changes the persistent observation history. All 80 queries share one
prefix encoding. The full-window control updates a disposable state copy with
the query; it also never persists hypothetical observations.

KDA evidence fusion compares `f + gate(f,m,e)*m` with the same-capacity generic
gate using zero evidence. GDN2 evidence revision supplies `e` to the existing
channel-wise erase/write projections, compared with zero evidence at exactly
the same parameter count. Here `e` is causal observed velocity innovation,
signed speed change and a validity bit, computed only from real prefix frames.
It is not a hidden intent, goal change timestamp or future truth.

There is one actor memory, not separate motion/context networks. Channel-wise
gates do not guarantee semantic motion/context separation or safe forgetting;
that is a hypothesis to test, not an architectural property already proved.

The compact cells implement the exact MIT FLA reference recurrence and omit
language-model convolutions, hybrid attention and large decoders. They do not
claim to reproduce the full Kimi Linear or GDN2 language-model architecture.
They need no additional CUDA package. Credit/license: vendor/FLA_LICENSE;
reference commit 9f38d24980c46d46bd38614e743cdacd21906578.

| Arm | Temporal/read interface | Parameters |
| --- | --- | ---: |
| actor_gru | Original actor-first GRU | 300,417 |
| gru_evidence | GRU prefix/read and evidence fusion | 366,593 |
| kda_full | Compact KDA with disposable query write | 268,177 |
| kda_read | KDA read-only query, current residual | 268,177 |
| kda_gate | KDA generic gated residual | 301,585 |
| kda_evidence | KDA evidence-gated residual | 301,585 |
| gdn2_read | GDN2 read, zero evidence at write gates | 335,241 |
| gdn2_revision | GDN2 evidence-conditioned write gates | 335,241 |

The two evidence-specific contrasts are parameter matched; comparisons between
different substrates are not. GRU evidence fusion is the strong cheap control.
Matrix-state capacity is also different: at these dimensions KDA/GDN2 store
40,960 floats versus the original GRU's 1,280, not a matched state-size control.
The frozen protocol uses seeds 191/223, the same immutable 128-episode ORCA
dataset, 50 IL epochs, 1,000 online MC episodes, four updates/episode, width128,
depth2, T24, reward/actions/simulator and 96 development cases/model. These
seeds are a new architecture pilot, not fresh confirmation of earlier trials.
Two seeds and reused cases cannot establish METHOD_ENTRY_FOUND.

```bash
python -m experiments.temporal_memory queue --data data/demonstrations.pt \
  --root outputs/memory_pilot --device cuda
python -m experiments.temporal_memory summarize --root outputs/memory_pilot
python -m experiments.temporal_memory latency --root outputs/memory_pilot \
  --seeds 191 --device cuda
python -m experiments.temporal_memory events --root outputs/memory_pilot \
  --seeds 191 --device cuda
```

Tests compare recurrence and gradients, official reference equations, causal
evidence, masks/re-entry, read-only candidate queries, shared-prefix versus
full-window values/gradients, and native candidate scores. Operator provenance
is not novelty: actor memory, separate current/history consumption and generic
gating have close priors, including ReCAT (https://intuitive-robots.github.io/ReCAT/).
TRACER (https://arxiv.org/html/2609.18776v1) also separates executed evidence
updates from candidate-trajectory queries in social navigation. That principle
is not a novel claim of this implementation.
Navigation results and evidence-specific ablations must justify any narrower
claim before the architecture is selected as a paper method.

The latency replay also measures the original trained actor GRU with a shared
prefix computation, preserving its value function. This prevents attributing
generic prefix reuse to a new memory operator. Natural-event shadow comparisons
use common roots from the first legal near-motion event in each pre-fixed parent
episode, not the best events for a new arm. They remain exploratory supporting
evidence, not a replacement for a negative paired SR result.

## Completed Memory Pilot

All eight arms finished both paired seeds (191/223): 16 final checkpoints,
50 IL epochs and 1,000 online MC episodes each, with 1,536 fixed evaluation
episodes in total. This cohort is separate from the older processing-order
experiments. No reward, action support, demonstration data or training budget
was changed after observing outcomes.

| Arm | SR | Collision | Timeout | Successful time (s) | RTX 4090 score (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| actor_gru | 82.81% | 6.25% | 10.94% | 18.20 | 3.013 |
| gru_evidence | 78.12% | 7.29% | 14.58% | 17.56 | 3.839 |
| kda_full | 74.48% | 15.62% | 9.90% | 19.51 | 11.678 |
| kda_read | 80.21% | 6.77% | 13.02% | 22.56 | 9.590 |
| kda_gate | 84.90% | 7.81% | 7.29% | 20.56 | 9.726 |
| kda_evidence | 75.52% | 6.25% | 18.23% | 23.47 | 9.716 |
| gdn2_read | 77.08% | 8.85% | 14.06% | 20.46 | 9.788 |
| gdn2_revision | 75.00% | 7.29% | 17.71% | 22.70 | 9.799 |

The parameter-matched mechanism tests are negative in both seeds:

- KDA evidence versus generic gate: SR -8.33 / -10.42 pp; mean -9.38 pp,
  timeout +10.94 pp. Adding motion evidence does not justify this gate.
- GDN2 evidence revision versus zero-evidence update: SR -1.04 / -3.13 pp;
  mean -2.08 pp, timeout +3.65 pp.
- Against the original actor GRU, the custom KDA/GDN2 arms lose 7.29 / 7.81 pp
  mean SR. Neither beats the GRU evidence control either.

Generic KDA gating has the highest mean SR, but its gain over actor GRU is only
+2.08 pp with one positive seed and one tie. Successful-episode time rises
about 13%; different success sets make this a descriptive, not causal, time
comparison. Its pooled 20-human SR is 76.56% versus 65.63% for actor GRU, but
this secondary reused-case slice does not rescue the failed primary gate or
establish a social-specific mechanism.

Every frozen contrast returns NO_CONSISTENT_PILOT_GAIN. This is
**VERSION_NEGATIVE, not FAMILY_NEGATIVE**; two seeds cannot establish permanent
dominance or a paper-ready method. No extra fresh training was launched.

### Cost and Validation

The timing table measures the complete 80-action score on an otherwise idle
RTX 4090, PyTorch 2.9.1+cu128, one CPU thread, 20 warmups and 100 synchronized
samples. The output-equivalent cached actor GRU takes 3.373 ms, so generic
prefix reuse is not a GPU speedup in this workload. KDA/GDN2 are roughly three
times slower than the original GRU here. These compact PyTorch cells are not
optimized official FLA kernels; this result does not benchmark those kernels.

On the i7-1165G7 laptop (PyTorch 2.4.1, one thread), actor GRU / cached GRU
take 64.032 / 5.530 ms. KDA evidence / GDN2 revision take 11.403 / 11.276 ms.
Thus the CPU caching benefit is already available without a new operator.
The local RTX 3060 replay is supplemental only: an unrelated RustDesk compute
process was active, so it is not an idle-device performance claim. Timings
across different devices/PyTorch versions are not pooled.

Training wall times per model are 573-778 s for the GRU arms and 1,446-2,063 s
for the matrix-memory arms. Varying concurrent worker counts and episode lengths
make these descriptive resource records, not matched throughput estimates.
Peak allocated memory is 692-724 MiB / 1,889-1,980 MiB respectively.

The common-root shadow covers 12 native parent episodes, 4,365 person-frames
and six first legal near-motion events. Over three-second continuations,
KDA evidence versus generic gate has three progress wins and three losses;
GDN2 revision versus its matched control has zero wins and four losses
(>=0.05 m). All branches are collision-free. Changed root actions therefore
do not establish recovery value or selective motion/context retention.

All 16 artifacts were checked for finite weights/losses, 50 IL epochs,
1,000 RL episodes, identical case sets and the shared data checksum. Source
and result/checkpoint/log hashes were compared with the training host. Normal
CLI loading was also checked for both custom checkpoints, not used as extra
performance evidence. The final local suite passes 50 tests, including legacy
Mamba parity and the NumPy-to-JSON shadow-export regression. The laptop runs
50 tests with 46 passing and four explicit optional-asset skips. All remote
artifacts were retrieved and checksum-verified before this run's temporary
4090 workspace was removed; existing environments were left untouched.

Full checkpoints and records remain local in outputs/memory_pilot, excluded
from Git. The existing strategy report contains the detailed paired contrasts.
The useful delivered result is a tested, compact architecture and reproducible
negative mechanism comparison, not a successful new navigation algorithm.

## Frozen KDA Gate Diagnostic

```bash
python -m experiments.temporal_memory gate-diagnostic \
  --root outputs/memory_pilot --device cuda
```

No new training: 38 common roots from 12 fixed native parent episodes, using
uniform ticks plus six first arrived near-motion events; both trained seeds.
KDA gating changes memory **readout**, not erase/write. The GDN2 update
mechanism is not tested by this read-gate diagnostic.

Removing only explicit motion evidence changes 0/76 candidate selections;
removing motion and validity changes 1/76. The direct mean gate change from
motion is about 0.00063. This does not support attributing the 9.38 pp SR gap
to harmful runtime motion gating on these states. Entire trained models differ,
and online MC refinement collects policy-dependent trajectories.

In the generic model, constant per-channel gates change 7/76 selections, a
uniform 0.5 gate changes 18/76, and no attenuation changes 35/76. Its mean gate
is 0.56, without broad saturation. This suggests readout scale calibration,
not demonstrated semantic stale-motion erasure. Constants use this same root
cohort; interventions are potentially out of distribution, final rankings
include the inherited safety filter, and no closed-loop improvement is claimed.
Raw diagnostics remain in outputs/memory_pilot/gate_diagnostic.json. A new
read-only intervention/restoration regression brings the local suite to 51
passing tests.

## Frozen Static-versus-Dynamic Follow-up

This follow-up trains read-only KDA with coefficient1, 128 learned
state-independent sigmoid channel scales, or the existing generic dynamic
gate. Actor GRU remains an external reference. Static scales initialize at0.5;
all shared KDA weights have identical initialization for a paired seed.
Parameters: 268,177 / 268,305 / 301,585; actor GRU has300,417. Capacity
differences are reported, not hidden using unused new parameters.

The separate frozen protocol uses four new seeds307/331/359/383 and cases
400-415 in circle/square with5/10/20 humans. Data, reward, actions,
50 IL epochs and1,000 MC-RL episodes are unchanged. Diagnostic IL50/RL500
snapshots are retained, but only the final checkpoint is eligible for the
primary comparison. Online trajectories still depend on the learned policy.

```bash
python -m experiments.temporal_memory queue \
  --protocol experiments/temporal_scale_protocol.json \
  --data data/demonstrations.pt --root outputs/scale_followup --device cuda
python -m experiments.temporal_memory summarize \
  --protocol experiments/temporal_scale_protocol.json --root outputs/scale_followup
```

Dynamic versus static is the primary contrast. A meaningful gain is at least
3 pp SR with3/4 positive seed pairs and the unchanged safety/progress limits.
Practical equivalence requires the paired90% t interval inside +/-3 pp for
aggregate SR only; failure to find a gain is not equivalence. The protocol was
frozen before any outcomes were inspected. Ordinary dynamic gating is not
automatically a new social-navigation mechanism.

### Four-seed Results (4 October 2026)

All16 models completed the frozen budget and1,536 evaluations. Only final
checkpoints are compared; neither intermediate snapshots nor the earlier
two-seed pilot are pooled into these results.

| Readout/reference | SR % | CR % | Timeout % | Successful time s | Successful path m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actor GRU | 79.17 | 10.16 | 10.68 | 16.62 | 11.69 |
| KDA read, coefficient1 | 76.82 | 11.72 | 11.46 | 22.21 | 16.12 |
| KDA static channel scale | 79.43 | 9.64 | 10.94 | 20.96 | 14.98 |
| KDA dynamic gate | 79.69 | 10.94 | 9.38 | 19.43 | 14.56 |

The primary dynamic-minus-static SR differences for307/331/359/383 are
-1.04 /0.00 /-7.29 /+9.38 pp. Mean +0.26 pp; paired90% interval
[-7.83,+8.35] pp. Only one positive pair, two negative and one tie:
**NO_CONSISTENT_PILOT_GAIN**, and practical SR equivalence is **not** established.
All five pre-fixed contrasts fail the pilot-gain rule. Dynamic-minus-GRU is
only +0.52 pp with one positive pair and16.90% longer successful time;
static-minus-GRU is +0.26 pp with26.09% longer successful time. Successful
time/path averages concern different surviving episode sets, not paired
progress equivalence. Six-cell supporting results remain in the raw summary.

| Complete80-action score | Idle4090 median ms | Laptop CPU median ms |
| --- | ---: | ---: |
| Actor GRU, original batched implementation | 3.02 | 56.43 |
| Actor GRU, mathematically equivalent prefix reuse | 3.37 | 5.38 |
| KDA read | 9.65 | 11.00 |
| KDA static | 9.65 | 11.07 |
| KDA dynamic | 9.74 | 11.47 |

These are100 repetitions after20 warmups, one pre-fixed five-human root,
T24 and no simulator/smoothing time. Server timing starts after all training
processes exit; CPU timing uses the laptop. KDA's apparent CPU advantage over
the unreused GRU is absorbed by prefix reuse; no efficiency advantage is found
over the stronger compute control. This compact recurrence is not the optimized
FLA kernel. KDA actor state is160 KiB versus5 KiB for GRU at this configuration.

Actual process training time is11.77-15.15 min for GRU,34.80-39.29 for KDA
read,30.06-36.32 for static and28.30-41.77 for dynamic. Concurrent load varies
from six to eight jobs; these are recorded costs, not isolated throughput
benchmarks. Summed overlapping training/evaluation times are7.94/0.79 process
hours, not GPU-hours. Peak allocated memory per training process is724 MiB
for GRU and1,890 MiB for KDA.

All48 checkpoints reload with exact configuration/parameter counts and finite
weights. Each log contains50 IL epochs and1,000 RL episodes; every model has
the same96 expected cases. The learned static coefficients finish near0.501,
with the full four-seed range0.4982-0.5051, so this control is close to uniform
attenuation rather than a strongly differentiated channel calibration.

**Interpretation:** the old two-seed dynamic-gate advantage does not replicate
as a stable gain here. This neither proves static/dynamic equivalence nor
rejects temporal navigation, actor memory or KDA as a family. It does not
support selective motion-evidence revision or a new method claim. Keep GRU
as the health/reference baseline. The next justified diagnosis is to locate
the divergence using retained IL50/RL500 snapshots under the same evaluator,
then test one identified replay/readout-contract issue; do not search hundreds
of outcome-selected gate variants or rescue a favorable seed.

Scientific source is frozen at6dde31e. Local results are in
/home/abc/workspace/shixu/outputs/scale_followup, including the protocol/source
manifest, paired summary, full episode records, learning logs, three checkpoints
per model and GPU/CPU latency arrays. Code is versioned; weights are not added
to Git. All98 remote raw artifacts and nine scientific source files match
local SHA256 checksums; laptop timing also matches its original checksum.
The server-only temporary workspace is removed after verification, with the
installed environment left intact. No additional training or architecture
is started by this analysis.

## Occlusion Research Loop

The occlusion experiments use legal measured/retained tracks, five-person
ORCA IL and online MC refinement, followed by reloaded-final-weight tests on
5/10/20 people in circle and square. The frozen primary endpoint is equally
weighted 10/20-person success, with collision, timeout and progress checks.
Both the ORCA teacher and the student receive legal measured/CV-retained
tracks from the same observation interface; never-seen pedestrians are excluded.
Temporary server outputs are copied locally before deletion. Development
results are not final evidence: a promising mechanism needs unseen seeds and
cases, with all controls retrained on the same host.

The V5 private-successor comparison completed all four seeds419/443/467/491.
Overall SR/CR/timeout are75.00/10.16/14.84% for measured-only KDA branches,
77.34/10.68/11.98% for CV-pseudowrite branches, and85.94/5.99/8.07% for the
strongest completed context-GRU reference. Their primary SRs are71.48/74.61/
85.55%. Measured-only branching loses to that reference in4/4 pairs. Neither
private branching nor excluding legal CV writes establishes a navigation gain.
Results remain in outputs/occlusion_v5; this is a version-level negative result.

A separate frozen-consumer shadow replaces only currently retained hidden
positions/velocities with current simulator truth, without introducing unseen
people or modifying tracker memory. Four-seed primary gain is only0.39 pp
(one positive, two negative, one tie). This is not a full-future upper bound:
it shows no large demonstrated hidden-state accuracy headroom for that frozen
consumer, not that temporal reasoning or occlusion handling is unnecessary.
Records remain in outputs/occlusion_v1/*/current/truth_retained.json.

V6 tests physical actor memory before current candidate geometry/goal fusion.
It uses the official-shaped KDA no-short-convolution mixer, verified against
the pinned FLA layer, rather than claiming that mixer or its output gate as new.
All motion-family models receive the same physical features and legal elapsed
interval input. Only the custom elapsed-clock arm scales channel log-decay by
the real interval; vanilla KDA uses unit decay per measurement. Controls include
same-placement GRU, CV pseudowrites, identical-capacity zero motion history,
current-only and the strong completed context-GRU. There is no extra loss or
prediction model. The clock mechanism is a hypothesis, not a demonstrated win.

```bash
python -m experiments.occlusion queue \
  --protocol experiments/occlusion_motion_protocol.json \
  --root outputs/occlusion_v6 --data outputs/occlusion_v6/demonstrations.pt \
  --arms motion_gru motion_kda motion_elapsed motion_imputed motion_nohistory \
  --device cuda --workers 2
```

Generic time-aware recurrence already exists in GRU-D and time-aware LSTM;
actor memory and delta-rule erase/write are also existing mechanisms. The
remaining question is whether legal observation-time semantics and this
physical/current fusion improve closed-loop navigation beyond those controls.
No novelty or safety guarantee is earned by passing numerical tests.

V6 scientific source is frozen at54d19cc. The 4090 runs PyTorch2.9.1/cu128,
and the3060 runs2.1.0/cu121; development comparisons retain this host boundary.
The CUDA packed-GRU test on2.9.1 differed from explicit stepping by2.36e-4
with default cuDNN TF32, and3.86e-6 with TF32 disabled. Float64 validates the
recurrence separately; production float32 kernels are not bitwise identical.
The archived optimized seed419 GRU replay changes controls in9/96 episodes
but changes no terminal outcomes. These checks cannot justify universal action
parity. Fresh confirmation must use one frozen implementation on one host.

### Physical-memory Results and the Next Readout Test

All20 V6 models completed50 IL epochs,1,000 online MC episodes and96
reloaded-final-weight evaluations each. Source, data, finite weights and the
complete episode keys are checked; all77 server artifacts match local checksums
before the temporary V6 workspace is removed.

| Model | Overall SR % | CR % | Timeout % | Primary10/20 SR % |
| --- | ---: | ---: | ---: | ---: |
| Current legal CV-track reference | 81.77 | 4.95 | 13.28 | 79.30 |
| Context-GRU reference | 85.94 | 5.99 | 8.07 | 85.55 |
| Physical-memory GRU | 87.50 | 4.43 | 8.07 | 83.59 |
| Physical KDA, unit observation clock | 80.99 | 3.65 | 15.36 | 76.56 |
| Physical KDA, elapsed clock | 80.99 | 2.86 | 16.15 | 76.95 |
| Physical KDA, CV pseudo-writes | 83.07 | 5.47 | 11.46 | 78.13 |
| KDA capacity control, zero committed history | 86.72 | 6.51 | 6.77 | 83.59 |

Elapsed-minus-unit primary differences are-9.38/+4.69/+7.81/-1.56 pp:
mean+0.39 pp, two positive and two negative. Elapsed-minus-physical-GRU is
-6.64 pp, with two losses and two ties. Fewer collisions are accompanied by
more timeouts, so this version does not meet the fixed success/progress rule.
The zero-history arm still receives legal interval/age input. These are
development results, not a family rejection or evidence that history is useless.

On836 uniformly sampled demonstration windows per checkpoint, first-layer
cross-actor key cosine averages0.297-0.327 across the four trained unit-clock
KDA models. Thus the earlier near-identical-address symptom is no longer
observed here; this statistic does not certify useful retrieval. Removing all
committed memory changes scalar values substantially, including hidden-track
windows, but value sensitivity does not establish better action ranking.

A separate frozen action diagnostic replays12 pre-fixed current-reference
episodes per seed and samples every eighth control step. Across626 states,
338 contain retained hidden actors. Zeroing only those actors' committed memory
changes79/338 actions (23.37%); zeroing all committed memory changes344/626
(54.95%). The value shift is mostly common across candidates, but the smaller
action-dependent component often changes the winner. This rules out a purely
common-offset explanation; it does not show that the changed actions are better.
The query geometry, age inputs, reward, action support and trained weights stay
fixed. All replayed terminal outcomes match the archived current-reference
episodes. Raw records are in outputs/occlusion_v6/action_content_*.json.

The corresponding frozen closed-loop shadow uses those first two cases per
cell (12 episodes per seed,48 total), without selecting strong interactions.
Original SR/CR/timeout are83.33/6.25/10.42%; masking only hidden-actor memory
gives79.17/2.08/18.75%, and masking all memory gives54.17/6.25/39.58%.
Hidden-memory removal reduces SR in two seeds, improves it in one and ties in
one. Memory matters to this trained model, with a safety/progress trade-off;
this intervention can be out of distribution and does not establish superiority
over a separately trained no-history model. Raw continuations and matched parent
records remain in outputs/occlusion_v6/closed_content_*_kda.json.

A separate four-seed CPU shadow checks the fixed2-second retention deadline.
Only previously seen expired actors still inside the legal history horizon
are eligible: current truth adds0.39 pp primary SR, while extending ordinary
CV retention adds0.78 pp. Of602 expired person-frames with a last measurement
still inside the legal prefix,31 are within2 m of the robot. This demonstrates
a lifecycle limitation but no large frozen-consumer headroom or residual beyond
the simple CV control. It does not justify another expiration architecture.
Raw records are in outputs/occlusion_v6/expiry_shadow.

V7 therefore replaces the readout interface, not another gate. Measured physical
actor streams write the same KDA memory; existing current robot/human fusion
moves before access and creates a different query for each candidate action.
The read-only arm contracts that query with the pre-existing matrix. It never
treats a hypothetical successor as a new measurement. A private-update KDA
control uses exactly the same fusion placement and parameters; a same-placement
GRU and a zero-committed-history control are also trained. No extra network,
prediction target, reward or data is introduced. Physical V6 references preserve
their original readout and budgets.

The mechanism hypothesis is that action-relevant retrieval can improve the
use of physical history beyond a candidate-independent actor summary. It is
not presumed true. The proposed read-before-write primitive is already present
in [DRAM](https://arxiv.org/abs/2609.32453), and task-conditioned retrieval is
not new. [Advantage-Driven Explicit Memory](https://arxiv.org/abs/2608.25610)
retrieves recurrent navigation experiences across episodes, unlike this bounded
per-actor measured stream. [Kimi Linear](https://arxiv.org/abs/2510.26692)
provides the mixer. A possible paper claim must concern the candidate-query/
occluded-actor interface and validated navigation benefit, not qS, KDA, actor
identity or generic read/write separation alone. Novelty remains unearned.

V7 is frozen at36d1702 with the same128 demonstrations, four paired seeds,
5-person training and5/10/20-person evaluation. Its pre-outcome protocol is
experiments/occlusion_query_protocol.json; results are in outputs/occlusion_v7.
All controls must remain in the comparison. Fresh seeds/cases remain untouched
until an eligible development winner is chosen.

These budgets complete the IL-to-online-MC pipeline, not the original camrl
training schedule:50 IL epochs use128 successful demonstrations, followed by
1,000 online episodes. At the final episode epsilon is still approximately
0.1335 on its1,500-episode decay schedule. They do not establish convergence.
If readout placement fails, a justified next test is one pre-fixed longer RL
budget for all competing arms, with the same demonstrations and architecture;
changing data volume and training budget together would not isolate the cause.

The pre-outcome budget protocol is experiments/occlusion_budget_protocol.json.
It fixes3,000 MC episodes for physical KDA, contextual GRU, same-placement GRU
and the original zero-committed-history KDA capacity control, keeping128 IL
demonstrations and50 IL epochs. No architecture changes are bundled with this
test. Matched final-IL checkpoints and all50 original IL log entries are reused
only after complete seed/configuration/hash checks; online replay is rebuilt
from the same demonstrations and all3,000 MC episodes are run, not resumed from
an incomplete RL buffer. The4090 handles419/443/491 and the3060 handles467.
Neither this budget nor the old1,000-episode budget is described as converged.
Fresh confirmation remains reserved.

### Candidate-query Results

All16 V7 models completed their fixed budgets and96 held-out episodes each.
All62 remote artifacts match local SHA256 checksums; final weights, complete
50-epoch/1,000-episode logs, shared data and episode keys are verified.

| Query interface | Overall SR % | CR % | Timeout % | Primary10/20 SR % |
| --- | ---: | ---: | ---: | ---: |
| Same-placement candidate-query GRU | 87.24 | 5.47 | 7.29 | 82.81 |
| KDA private candidate update | 71.61 | 8.07 | 20.31 | 70.31 |
| KDA read-only candidate query | 75.78 | 9.90 | 14.32 | 74.61 |
| Read-only query, zero committed history | 81.25 | 5.99 | 12.76 | 77.34 |

Read-only-minus-same-placement-GRU primary differences are
-26.56/-4.69/0.00/-1.56 pp: three losses and one tie. Against contextual GRU,
the mean is-10.94 pp, with4.69 pp more collisions and6.25 pp more timeouts.
Read-only access beats the private KDA update on aggregate SR, but not the
pre-fixed safety/seed-consistency rules or the stronger controls. This version
does not establish better retrieval, navigation or a new method. Its result is
VERSION_REQUIRES_DIAGNOSIS_NOT_DIRECTION_REJECTED; results and checks remain in
outputs/occlusion_v7. V8 isolates the learning-budget question rather than
adding another query gate or selecting a favorable V7 seed.

### Query-contract Check During the Budget Run

An independent held-out collection contains64 successful legal-observation
ORCA episodes from cases13000 onward. Final-IL snapshots, not final-RL weights,
are compared against expert continuation returns on762 uniformly sampled
transitions;48 active-support changes are excluded before model evaluation.
Each transition has the same legal prefix and target. Only the final query
changes from the actual next observation to the planner's CV successor.

| Four-seed mean MSE | Observed next | CV next | CV with fresh-age shadow |
| --- | ---: | ---: | ---: |
| Context GRU | 0.00230110 | 0.00231515 | 0.00232848 |
| Physical GRU | 0.00219292 | 0.00219781 | 0.00220714 |
| Physical KDA | 0.00240531 | 0.00241053 | 0.00241885 |

The age shadow resets only the candidate age of actors measured at the root;
it is fictitious and is not a proposed deployment fix. These results do not
show a substantial aggregate query-mismatch penalty in this expert cohort.
They therefore do not justify treating that mismatch as the demonstrated cause
of poor navigation, or changing the training target on that basis. This check
does not cover on-policy RL errors,20-person observations or action quality.
The helper is experiments/query_contract_probe.py; raw predictions, data hashes
and checkpoint hashes remain in outputs/query_contract.

V8 used one frozen source for every arm. The resumed seed467 physical-GRU
prefix exactly matches its earlier terminal outcomes and losses over706 checked
episodes. The older context-GRU run used a different but semantically equivalent
packed-GRU implementation, so its resumed prefix is not an exact replay. Any
old-versus-new context-GRU difference cannot be attributed exclusively to budget;
the current V8 within-run comparison remains matched. Fresh confirmation, if
earned, must train all arms from scratch on one frozen implementation and host.

### Budget Results and Round Closure

All16 V8 models completed50 IL epochs and3,000 online MC episodes, then96
final-weight tests each. All95 remote artifacts match local SHA256 checksums;
the server RAM workspace is removed. Complete records are in outputs/occlusion_v8.

| Model | Overall SR % | CR % | Timeout % | 5-person SR % | 10-person SR % | 20-person SR % | Primary10/20 SR % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Context GRU | 85.42 | 6.51 | 8.07 | 93.75 | 84.38 | 78.12 | 81.25 |
| Same-placement physical GRU | 86.98 | 6.25 | 6.77 | 98.44 | 86.72 | 75.78 | 81.25 |
| Physical KDA | 86.98 | 5.99 | 7.03 | 98.44 | 87.50 | 75.00 | 81.25 |
| Zero-committed-history KDA | 87.76 | 7.55 | 4.69 | 94.53 | 92.19 | 76.56 | 84.38 |

KDA-minus-context primary differences are-9.38/-7.81/-6.25/+23.44 pp;
KDA-minus-physical-GRU differences are-4.69/-6.25/+6.25/+4.69 pp. Equal
pooled rates are not an equivalence result. Against zero committed history,
KDA loses3.125 pp primary SR and incurs5.08 pp more timeouts while reducing
collisions1.95 pp. No contrast meets the pre-fixed admission rules.

The KDA1,000-episode prefix exactly reproduces V6 for all four seeds. Its
primary SR rises4.69 pp with the larger budget, but only2/4 paired seeds rise;
overall successes increase311 to334, collisions14 to23, timeouts59 to27.
This supports a partial budget effect, not convergence or a stable advantage.

No fresh confirmation or V9/V10 is started. The current version is stopped,
not the entire temporal-model family rejected. The Chinese interim report
OCCLUSION_KDA_INTERIM_REPORT_20261004.md records the final results, evidence
boundaries and diagnostic gaps. No associative-memory necessity, learned
selective revision, or MC-objective bottleneck is established by these results.

### Read-only Closure Audit

The user's five diagnostic questions are answered in section14 of the Chinese
report. No new training, controller or reward change is part of this audit.

```bash
PYTHONPATH=vendor:. python -m experiments.ranking_audit \
  --root outputs/occlusion_v8 --old-root outputs/occlusion_v6 \
  --expert-data outputs/query_contract/demonstrations64.pt \
  --output outputs/query_contract/closure_ranking_audit.json
```

Sixteen failure trajectories were selected by a fixed first-key rule; eight
timeouts and six sufficiently long collisions supply55 shared frozen states.
Both physical models and their IL50/RL1000/RL3000 snapshots receive identical
legal histories and80 candidates. Every source final-weight executed command
matches its archive. The independent true-human two-second geometry reference
holds each candidate velocity constant; it is not an optimal Q or a closed-loop
navigation improvement. Different-time outcomes are not pooled as one benchmark.

Of40 timeout anchors,32 admit a one-step CV-safe-progress action. Ranking by
scalar V selects one in15 GRU versus3 KDA states; native reward-plus-value
selects20 versus8, filtering27 versus15, smoothing20 versus13. This locates
part of the difference before filtering, not specifically inside memory read.
Under a stricter true two-second clearance/progress criterion, native ranking
selects5/20 for GRU versus8/20 for KDA: the contrary evidence rules out calling
KDA's short-term ranking uniformly worse. Pre-collision5–10-second coverage is
insufficient to establish a gradually forming bias.

Final-IL held-out expert-return MSE is0.00219292 for GRU and0.00240531 for KDA.
Post-RL discrepancies against those expert returns are not on-policy value
errors; no archived common online replay or true80-action value reference
supports declaring the MC objective the cause. Earlier RL500 snapshots exist
only for a different full-observation cohort and cannot fill this curve.
The current V8 query uses common physical human successors; candidate robot
actions enter post-read fusion, not KDA q. Learned retrieval semantics and
associative-memory necessity remain unproved. Current tests:91,88 pass and
three optional legacy-asset skips. Main frozen scientific sources are unchanged.


<!-- END PRESERVED SOURCE -->
