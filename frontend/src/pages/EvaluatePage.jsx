import React, { useEffect, useState, useCallback } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    Cell, LabelList
} from 'recharts';
import {
    Play,
    RefreshCw,
    TrendingUp,
    Target,
    Layers,
    Info,
    AlertCircle,
    Zap,
    AlertTriangle,
    ImageOff
} from 'lucide-react';
import useImageStore from '../store/useImageStore';

const API_BASE_URL = 'http://localhost:8000';

// ───────────────────────────────────────────────────────────────────────────
// Worst Queries Section
// ───────────────────────────────────────────────────────────────────────────
// ───────────────────────────────────────────────────────────────────────────
// Performance Queries Section (Best/Worst)
// ───────────────────────────────────────────────────────────────────────────
const PerformanceSection = ({ items = [], title, icon: Icon, thresholdLabel, type = "worst" }) => {
    const barColor = (map5) => {
        if (map5 === 0) return '#ef4444';
        if (map5 < 0.4) return '#f97316';
        if (map5 < 0.7) return '#eab308';
        return '#22c55e';
    };

    if (!items || items.length === 0) return null;

    return (
        <div className="bg-slate-800/40 p-6 rounded-2xl border border-slate-700/50 shadow-xl h-full">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-base font-black text-white flex items-center gap-2">
                    <Icon className={`w-4 h-4 ${type === 'worst' ? 'text-orange-400' : 'text-emerald-400'}`} />
                    {title}
                    <span className="text-[10px] font-medium text-slate-500 bg-slate-900/60 px-2 py-0.5 rounded-full border border-slate-700/50 ml-1">
                        {thresholdLabel}
                    </span>
                </h3>
            </div>

            <div className="space-y-3 max-h-[800px] overflow-y-auto pr-2 custom-scrollbar">
                {items.map((item) => {
                    const imgUrl = item.file_path ? `${API_BASE_URL}${item.file_path}` : null;
                    const pct = Math.round(item.map5 * 100);
                    const color = barColor(item.map5);

                    return (
                        <div key={item.image_id} className="flex items-center gap-4 bg-slate-900/50 rounded-xl p-3 border border-slate-700/30 hover:border-slate-600/50 transition-all group">
                            <div className="w-8 h-8 flex-shrink-0 flex items-center justify-center rounded-lg bg-slate-800 border border-slate-700 text-slate-400 font-black text-xs">
                                #{item.rank}
                            </div>

                            <div className="w-12 h-12 flex-shrink-0 rounded-lg overflow-hidden bg-slate-800 border border-slate-700/50">
                                {imgUrl ? (
                                    <img src={imgUrl} alt={item.file_name} className="w-full h-full object-cover" />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center text-slate-600"><ImageOff className="w-4 h-4" /></div>
                                )}
                            </div>

                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-semibold text-slate-200 truncate group-hover:text-white transition-colors">{item.file_name}</p>
                                <div className="mt-1 flex items-center gap-2">
                                    <div className="flex-1 bg-slate-700/50 rounded-full h-1 overflow-hidden">
                                        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
                                    </div>
                                    <span className="text-[10px] font-black w-8 text-right" style={{ color }}>{pct}%</span>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

// ───────────────────────────────────────────────────────────────────────────
// Ground Truth Stats Section
// ───────────────────────────────────────────────────────────────────────────
const GroundTruthStats = ({ stats }) => {
    if (!stats || !stats.clusters) return null;

    const data = [...stats.clusters].sort((a, b) => b.count - a.count);

    return (
        <div className="bg-slate-800/40 p-8 rounded-3xl border border-slate-700/50 shadow-xl">
            <h3 className="text-sm font-black text-slate-400 uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-400" />
                Ground Truth Distribution
            </h3>
            <div className="w-full" style={{ height: `${Math.max(200, data.length * 30)}px` }}>
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} layout="vertical" margin={{ left: 100, right: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                        <XAxis type="number" hide />
                        <YAxis 
                            dataKey="name" 
                            type="category" 
                            stroke="#94a3b8" 
                            fontSize={10} 
                            width={90}
                            tickFormatter={(val) => val.length > 15 ? val.substring(0, 12) + '...' : val}
                        />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                        />
                        <Bar dataKey="count" fill="#10b981" radius={[0, 4, 4, 0]} barSize={20}>
                            <LabelList dataKey="count" position="right" fill="#94a3b8" fontSize={10} offset={10} />
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fillOpacity={1 - (index * 0.02)} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

// ───────────────────────────────────────────────────────────────────────────
// Main EvaluatePage
// ───────────────────────────────────────────────────────────────────────────
const EvaluatePage = () => {
    const {
        evaluationData,
        evaluationLoading,
        fetchEvaluation,
        triggerOptimization,
        isOptimizing,
        setIsOptimizing
    } = useImageStore();

    const [trials, setTrials] = useState(50);

    useEffect(() => {
        fetchEvaluation();
    }, [fetchEvaluation]);

    const handleOptimize = async () => {
        await triggerOptimization(trials);
    };

    useEffect(() => {
        let interval;
        if (isOptimizing) {
            interval = setInterval(() => {
                fetchEvaluation(true); // Silent update
            }, 2000);
        }
        return () => {
            if (interval) clearInterval(interval);
        };
    }, [isOptimizing, fetchEvaluation]);

    const ModelEvaluationSection = ({ data, isOptimizing }) => {
        if (!data) return null;

        const chartData = data.weights
            ? Object.entries(data.weights)
                .map(([name, value]) => ({ name, value: parseFloat((value * 100).toFixed(2)) }))
                .sort((a, b) => b.value - a.value)
                .filter(item => item.value > 0.5)
            : [];

        const prCurveUrl = `${API_BASE_URL}/static/visualizations/pr_curve_optimized.png`;
        const pkCurveUrl = `${API_BASE_URL}/static/visualizations/pk_curve_optimized.png`;
        const scoreDistUrl = `${API_BASE_URL}/static/visualizations/score_dist_optimized.png`;
        const sparsityUrl = `${API_BASE_URL}/static/visualizations/sparsity_vs_accuracy_optimized.png`;

        return (
            <div className="space-y-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Left Column: Weights & Sparsity */}
                    <div className="space-y-8">
                        <div className="bg-slate-800/40 p-8 rounded-2xl border border-slate-700/50 shadow-xl h-fit">
                            <h3 className="text-xl font-bold text-white mb-8 flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <Layers className="w-5 h-5 text-primary-500" />
                                    Optimized Weights (%)
                                </div>
                                <span className="text-xs font-medium text-slate-500 bg-slate-900/50 px-3 py-1 rounded-full border border-slate-700/50">
                                    {chartData.length} active features
                                </span>
                            </h3>
                            <div className="w-full" style={{ height: `${Math.max(500, chartData.length * 18)}px` }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 60, left: 80, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                                        <XAxis type="number" hide />
                                        <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={9} width={70} interval={0} />
                                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                                        <Bar 
                                            dataKey="value" 
                                            fill="#3b82f6" 
                                            radius={[0, 4, 4, 0]}
                                            isAnimationActive={!isOptimizing}
                                        >
                                            <LabelList dataKey="value" position="right" fill="#94a3b8" fontSize={11} formatter={(v) => `${v}%`} />
                                            {chartData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={index < 3 ? '#6366f1' : '#3b82f6'} fillOpacity={1 - (index * 0.05)} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        <div className="bg-slate-800/40 p-6 rounded-2xl border border-slate-700/50 shadow-xl">
                            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                                <TrendingUp className="w-3 h-3 text-primary-500" /> Sparsity vs Accuracy (Pareto Frontier)
                            </h4>
                            <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/30">
                                <img src={`${sparsityUrl}?t=${data.timestamp}`} alt="Sparsity vs Accuracy" className="w-full object-contain" />
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Performance Charts */}
                    <div className="space-y-6">
                        <div className="bg-slate-800/40 p-6 rounded-2xl border border-slate-700/50 shadow-xl">
                            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                                <TrendingUp className="w-3 h-3 text-primary-500" /> Precision-Recall Analysis
                            </h4>
                            <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/30">
                                <img src={`${prCurveUrl}?t=${data.timestamp}`} alt="PR Curve" className="w-full object-contain" />
                            </div>
                        </div>

                        <div className="bg-slate-800/40 p-6 rounded-2xl border border-slate-700/50 shadow-xl">
                            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                                <Target className="w-3 h-3 text-primary-500" /> Precision at K (1-20)
                            </h4>
                            <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/30">
                                <img src={`${pkCurveUrl}?t=${data.timestamp}`} alt="P@K Curve" className="w-full object-contain" />
                            </div>
                        </div>

                        <div className="bg-slate-800/40 p-6 rounded-2xl border border-slate-700/50 shadow-xl">
                            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                                <Zap className="w-3 h-3 text-primary-500" /> Score Separation (GT vs Others)
                            </h4>
                            <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/30">
                                <img src={`${scoreDistUrl}?t=${data.timestamp}`} alt="Score Separation" className="w-full object-contain" />
                            </div>
                        </div>
                    </div>
                </div>

                <GroundTruthStats stats={data.gt_stats} />

                {/* ── Query Performance Analysis ── */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <PerformanceSection 
                        items={data.worst_queries} 
                        title="Underperforming Queries"
                        icon={AlertTriangle}
                        thresholdLabel="mAP@5 < 50%"
                        type="worst"
                    />
                    <PerformanceSection 
                        items={data.best_queries} 
                        title="Top Performing Queries"
                        icon={TrendingUp}
                        thresholdLabel="mAP@5 ≥ 50%"
                        type="best"
                    />
                </div>
            </div>
        );
    };

    if (evaluationLoading && !evaluationData) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[60vh] text-slate-400">
                <RefreshCw className="w-12 h-12 mb-4 animate-spin text-primary-500" />
                <p className="text-lg animate-pulse">Analyzing multi-model performance...</p>
            </div>
        );
    }

    const hasData = evaluationData !== null;

    return (
        <div className="max-w-7xl mx-auto space-y-8 pb-12">
            {/* Unified Control Center & Metrics Row (50/50 Split) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
                {/* Left: Control Center */}
                <div className="bg-slate-800/40 backdrop-blur-md p-6 rounded-3xl border border-slate-700/50 shadow-2xl flex items-center justify-between gap-6">
                    <div className="flex items-center gap-6">
                        <div className="space-y-1">
                            <p className="text-[9px] text-slate-500 uppercase font-black tracking-widest mb-1">Trials</p>
                            <input
                                type="number" value={trials} onChange={(e) => setTrials(parseInt(e.target.value))}
                                className="bg-slate-900 border border-slate-700 text-white text-sm rounded-xl focus:ring-primary-500 block w-20 p-2"
                            />
                        </div>
                    </div>

                    <div className="flex gap-2">
                        {!isOptimizing ? (
                            <button
                                onClick={handleOptimize}
                                className="flex items-center gap-3 px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white rounded-xl font-black shadow-xl shadow-primary-600/20 transition-all hover:-translate-y-0.5"
                            >
                                <Play className="w-4 h-4 fill-current" />
                                START
                            </button>
                        ) : (
                            <button
                                onClick={() => setIsOptimizing(false)}
                                className="flex items-center gap-3 px-6 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-500 border border-red-500/30 rounded-xl font-black transition-all"
                            >
                                <RefreshCw className="w-4 h-4 animate-spin" />
                                STOP
                            </button>
                        )}
                        <button
                            onClick={fetchEvaluation}
                            className="w-[44px] h-[44px] flex items-center justify-center rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-400 border border-slate-700 transition-all aspect-square"
                        >
                            <RefreshCw className={`w-4 h-4 ${evaluationLoading ? 'animate-spin' : ''}`} />
                        </button>
                    </div>
                </div>

                {/* Right: Metrics Dashboard */}
                <div className="bg-slate-800/40 backdrop-blur-md p-6 rounded-3xl border border-slate-700/50 shadow-2xl flex items-center justify-around gap-4">
                    {hasData ? (
                        <>
                            <div className="text-center px-4">
                                <span className="text-primary-400 font-bold uppercase text-[9px] tracking-widest block mb-1">mAP Total</span>
                                <span className="text-2xl font-black text-white">{((evaluationData.metrics?.test_map_after || 0) * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-10 w-px bg-slate-700/50"></div>
                            <div className="text-center px-4">
                                <span className="text-primary-400 font-bold uppercase text-[9px] tracking-widest block mb-1">mAP@5</span>
                                <span className="text-2xl font-black text-white">{((evaluationData.metrics?.test_map5_after || 0) * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-10 w-px bg-slate-700/50"></div>
                            <div className="text-center px-4">
                                <span className="text-primary-400 font-bold uppercase text-[9px] tracking-widest block mb-1">mAP@10</span>
                                <span className="text-2xl font-black text-white">{((evaluationData.metrics?.test_map10_after || 0) * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-10 w-px bg-slate-700/50"></div>
                            <div className="text-center px-4">
                                <span className="text-emerald-400 font-bold uppercase text-[9px] tracking-widest block mb-1">GT Images</span>
                                <span className="text-2xl font-black text-white">{evaluationData.gt_stats?.total_images || 0}</span>
                            </div>
                        </>
                    ) : (
                        <div className="flex items-center gap-3 text-slate-500 font-bold italic text-sm">
                            <Info className="w-5 h-5 text-primary-500" />
                            Run optimization to see performance metrics.
                        </div>
                    )}
                </div>
            </div>

            {!hasData ? (
                <div className="bg-slate-800/20 border-2 border-dashed border-slate-700/50 p-20 rounded-3xl text-center">
                    <AlertCircle className="w-16 h-16 text-slate-600 mx-auto mb-6" />
                    <h3 className="text-2xl font-bold text-slate-400">No Optimization Results Yet</h3>
                    <p className="text-slate-500 mt-2">Run the optimization process to generate performance data based on your folder structure labels.</p>
                </div>
            ) : (
                <ModelEvaluationSection data={evaluationData} isOptimizing={isOptimizing} />
            )}
        </div>
    );
};

export default EvaluatePage;
