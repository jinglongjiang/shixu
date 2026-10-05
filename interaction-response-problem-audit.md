# Interaction-Response Temporal Problem Audit

日期：2026-10-05。范围：只查SICNav/TRACER这两条既有线索，不训练、不修改PaS、不追加KDA、不开发新消费者。

## 1. 结论

**本轮没有找到可以进入新时序算法开发的、已验证动作收益的原生交互响应问题。暂停当前“新时序模型做导航”的模型研发，不自动扩搜索。**

不能把三件事混在一起：

1. 人会响应机器人：SICNav源码与本轮函数测试均确认。
2. 响应还依赖当前状态之外的持续交互状态：当前SICNav默认ORCA没有发现这个能力；TRACER以作者定义的持续模式协议研究过它，但不是我们的本地验证。
3. 合法历史能改善真实导航动作，并留下新方法空间：本轮没有这条证据。

上一轮PaS已封账：长CV独有挽救0/8、保护集8/8完成，长hold有不满足安全要求的挽救。保持pas-failure-confirmation.md的原结果。负结果支持停止具体用途，不能扩大为所有物理状态记忆或所有时序导航问题都已被否定。

## 2. 原生任务与资产核查

| 入口 | 本地/文献资产 | 任务变化 | 本轮判定 |
|---|---|---|---|
| SICNav / CrowdSimPlus | c702fb8源码、已安装sicnav环境、两个既有SICNav-np运行记录 | 机器人可见、原生人机响应、静态墙/瓶颈、unicycle MPC；不是shixu的80动作IL-MC | 即时交互可运行；没有所需的持续响应模式证据 |
| SICNav-Diffusion | 同仓库有JMID/iMID权重 | 学习预测与MPC的另一消费接口 | 仅资产核查，不转开旧预测桥梁，不运行新权重实验 |
| TRACER | 原论文及公式/消融；未核得官方代码+环境+checkpoint可运行组合 | 多机器人、持续响应mode及执行证据更新 | 有先例线索，复现资产阻塞；不能当作本地新发现 |

SICNav官方库确实提供ORCA/SFM与交互规划，不把“有官方代码”当成额外历史能力已经验证。[官方仓库](https://github.com/sepsamavi/safe-interactive-crowdnav)

本地temp副本已有他人/旧任务对CrowdSimPlus的修改：square生成器补env引用，discomfort_dist改读reward字典。未回退、未覆盖；这次使用hallway/circle，不新造噪声、遮挡或行为模式。协议记录脏状态、相关源码和旧资产哈希。

## 3. SICNav具体缺不缺跨时刻响应状态

以下是本地源码事实，而不是“理论上应该需要记忆”：

- crowd_sim_plus/envs/crowd_sim_plus.py:1043–1054：给每个人当前其他人的观测；robot.visible为true时，加入当前机器人位置、速度、半径。
- crowd_sim_plus/envs/utils/human_plus.py:105起：Human.act将自己的FullState、当前邻居、静态墙交给policy。自己的目标和v_pref仍参与环境生成，未作为新增可部署输入。
- crowd_sim_plus/envs/policy/orca_plus.py:55–71：以当前p/v/r、目标、v_pref构建ORCA并计算当前preferred velocity。
- 同文件:84–90：完成doStep后只记录last_state，并将sim置None；last_state没有在下一次响应计算中读取。没有“刚才被这个机器人逼近/让行后，持续改变某个响应参数”的更新。
- SFM备选: social_force.py:39起：当前p/v、目标与固定力参数决定更新；is_bottleneck来自场景，不是由执行交互学到的响应mode。本轮没有执行SFM对照。
- sicnav/policy/campc.py:1285–1315：非特权臂用当前p+2v估计目标。求解器warm start/上一解缓存是规划器状态，不是人的持续响应状态。

因此，在固定人目标、v_pref、邻居、墙和模型参数的完整当前状态下，默认ORCA响应不额外读取过去交互。**这不意味着机器人合法当前观测充分**：人的目标/v_pref仍可能隐藏。那属于另一种隐状态恢复问题，不能重新包装成已发现的持续让行响应记忆。

## 4. 有限回放与kill test

仅复用两个已有成功运行的SICNav-np case0记录，没有新episode、MPC solve或新导航策略：

| 既有配置 | 日志控制步 | 预先固定root tick | 当前物理状态最大回放偏差 |
|---|---:|---|---:|
| hallway_bottleneck，3人 | 15 | 0、7、14 | 1.11e-16 |
| circle_crossing，5人 | 18 | 0、9、17 | 1.85e-13 |

旧pkl包含终局摘要及优化x/u，不含完整原观测、真正执行命令、原奖励或响应belief。用保存u的首列，按原ActionRot的omega乘dt转换回放；每步将重建p/v/heading/signed-speed与保存x的当前列核对，容差事先固定1e-5。33步全部通过。

这提供有限的物理状态回放，不等于原完整训练/评估复现。保存x中未来列与估计目标没有冒充真实后果标签；两个成功记录也没有被说成自然失败cohort。

kill test：在6个预定root上，对每个人保持相同完整当前状态、目标、邻居、机器人及墙，比较已承接原生运行的ORCAPlus与相同参数的全新ORCAPlus。只重置人的policy缓存，不改变世界或消费者。

| 测试 | 结果 | 能说明什么 |
|---|---|---|
| 24次warm vs fresh响应查询 | **0/24改变，输出逐元素精确一致** | 当前默认ORCA缓存没有额外响应记忆作用，支持源码检查 |
| 函数正控制：只改变当前机器人速度 | **10/24改变** | 人确实消费当前机器人运动，不是“完全无交互” |
| 自然同几何、异历史、异响应失败配对 | **没有建立** | 不能宣布已找到时序决策问题 |
| 导航动作/Q/终局改善 | **没有测得/没有声称** | 上述是human-response函数测试，不是方法收益 |

正控制仅在函数输入中将当前robot velocity设为沿朝向的正/负v_pref；未执行这些速度，未修改benchmark可见性，也不是机器人动作最优性测试。

没有用近邻配对制造历史优势，也没有隐瞒当前goal/v_pref的差异：这些量在warm/cold比较中完全相同。这个kill test否证的是**默认ORCA中额外跨时刻响应缓存**，不是“人的真实社会行为没有历史”。

## 5. TRACER近邻核查与反证

TRACER III-C以执行动作和同步观察更新identity-bound模式belief；IV-A的持续模式试验由作者设置固定跨窗口mode，IV-C/D报告其预测及决策作用。这说明有相关研究先例，不是本地原生问题确认。未核得可直接运行的官方资产组合，不为复现其结论自行注入mode或重建系统。[TRACER原论文](https://arxiv.org/html/2609.18776v1)

**其外部SocialGym2结果不支持把完整系统收益归因于持续belief：** Table III(b)的完整TRACER完成率31.6%、碰撞率30.3%；Reset为32.2%、29.7%；NoBelief为32.2%、29.4%。完整版本相对GoAlone改善，不等于相对相同系统的记忆消融改善。没有原始配对数据或区间，不能把小差异说成显著劣势；也不能将合成模式结果外推成外部任务中的稳定时序增量。[Table III(b)](https://arxiv.org/html/2609.18776v1#S4.T3)

这是对早前“该先例已支持持续交互历史带来可迁移导航价值”解释的收紧。通用累计响应belief和identity一致更新已有覆盖；相似工作不自动拒绝方向，但当前没有核出我们的具体新增能力和实际残差。

## 6. 当前状态—缺失变量—历史—动作链

| 链条 | SICNav默认ORCA | TRACER线索 |
|---|---|---|
| 当前合法状态不充分 | 目标/v_pref可能隐藏；未证明新的响应模式歧义 | 作者定义latent mode，属于其任务设定 |
| 缺失的交互响应变量z | 没有找到由过去robot interaction持续更新的z | 明确有模式belief，但已经是其方法 |
| 合法历史可恢复z | 本轮未建立；warm/cold没有响应差异 | 作者受控实验有报告，本地未复现 |
| 改变合理导航动作并改善后果 | 没有本地证据 | 外部Reset/NoBelief汇总未显示完整版本增量 |
| 简单方法剩余缺口 | 未建立，不预设GRU失败 | 普通离散Bayes/累计统计是强替代；无本地比较 |

**没有一条线完成本轮要求的四段链条，因此不进入Top3或算法阶段。** 不把资产阻塞说成科学问题被否定，也不因“人机交互常见”自动升级。

## 7. 决策与重启条件

已证实：所查SICNav原生支持当前人机响应；24次固定完整状态的缓存重置没有改变人的响应；旧33步物理回放通过；TRACER已包含相关机制，其外部记忆消融不支持稳定增量主张。

有证据支持的判断：在这些既有资产上继续迁移KDA/GRU或人为注入社会响应mode，没有充分研发依据。暂停当前新时序模型路线。

尚未证明：真实人机历史一般无价值；其他原生交互任务不存在持续模式；没有人能在同类问题形成新贡献。这些不能由本轮结果推断。

唯一下一步：**归档后停止模型投入和自动搜索。** 重启至少需要原生任务的可运行持续响应资产，以及“当前信息匹配、合法执行历史能区分响应、真实动作后果改善”的证据；不要求GRU先失败才承认问题存在。还须单独说明相对既有累计belief/简单滤波的具体研究空间。

## 8. 复现资产

入口：experiments/interaction_response_audit.py。使用现有sicnav环境；没有改母体、权重、奖励、训练或规划器。

```bash
/home/abc/miniconda3/envs/sicnav/bin/python -m experiments.interaction_response_audit
```

脚本拒绝覆盖protocol/results。协议保存两份旧运行、9份母体代码/配置及诊断源码哈希，运行结束重新核查未变。首次临时探针缺少官方policy_factory注册导致KeyError，未产生结果；补正确官方导入后运行，未修改源码或dependency。

protocol SHA256：830fb3307c001d9b4272478c1b2483a642b63c9faa07be20f3c304b5d4563a63。

results SHA256：5e11a40d5007714d81a01aa77738097eac889f523052d9fe0db695776b40da36。归档前再次核对协议、结果、诊断源码、9份母体代码/配置和两份旧运行记录的哈希，全部一致。协议与24条原始查询结果一并纳入Git；外部依赖和旧pkl仍保留本地，不声称单独clone shixu即可完整复现。

结果目录：

/home/abc/workspace/shixu/outputs/interaction_response_audit/

新测试核对current列与未来/goal列隔离、signed speed、冻结结果禁止覆盖。所有本轮数字来自results.json；没有新导航checkpoint和收益报告。

完整仓库回归：181 passed，3 skipped，68 warnings；三个legacy parity测试因未设置原源码/checkpoint路径而跳过，警告来自现有NumPy/SciPy兼容性。执行命令为env PYTHONPATH=vendor:. /usr/bin/python -m pytest -q -rs tests。本轮未修改模型、训练、动作执行或母体源代码。
