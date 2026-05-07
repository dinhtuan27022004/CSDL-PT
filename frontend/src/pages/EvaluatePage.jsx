import React, { useEffect, useState } from 'react';
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
    Zap
} from 'lucide-react';
import useImageStore from '../store/useImageStore';

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
    const [allowNegative, setAllowNegative] = useState(false);

    useEffect(() => {
        fetchEvaluation();
    }, [fetchEvaluation]);

    const handleOptimize = async () => {
        await triggerOptimization(trials, allowNegative);
    };

    useEffect(() => {
        let interval;
        if (isOptimizing) {
            interval = setInterval(() => {
                fetchEvaluation();
            }, 5000);
        }
        return () => {
            if (interval) clearInterval(interval);
        };
    }, [isOptimizing, fetchEvaluation]);

    const API_BASE_URL = 'http://localhost:8000';

    const ModelEvaluationSection = ({ data }) => {
        if (!data) return null;

        const chartData = data.weights ?
            Object.entries(data.weights)
                .map(([name, value]) => ({ name, value: parseFloat((value * 100).toFixed(2)) }))
                .sort((a, b) => b.value - a.value)
                .filter(item => item.value > 0.5)
            : [];

        const prCurveUrl = `${API_BASE_URL}/static/visualizations/pr_curve_optimized.png`;
        const pkCurveUrl = `${API_BASE_URL}/static/visualizations/pk_curve_optimized.png`;
        const scoreDistUrl = `${API_BASE_URL}/static/visualizations/score_dist_optimized.png`;
        const sparsityUrl = `${API_BASE_URL}/static/visualizations/sparsity_vs_accuracy_optimized.png`;

        return (
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
                                    <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]}>
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
                            <input
                                type="number" value={trials} onChange={(e) => setTrials(parseInt(e.target.value))}
                                className="bg-slate-900 border border-slate-700 text-white text-sm rounded-xl focus:ring-primary-500 block w-20 p-2"
                            />
                        </div>

                        <div className="flex items-center gap-3">
                            <label htmlFor="neg" className="relative inline-flex items-center cursor-pointer group">
                                <input
                                    type="checkbox" id="neg" checked={allowNegative}
                                    onChange={(e) => setAllowNegative(e.target.checked)}
                                    className="sr-only peer"
                                />
                                <div className="w-10 h-5 bg-slate-700 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary-600"></div>
                                <span className="ml-2 text-xs font-bold text-slate-400 group-hover:text-slate-300">Neg Weights</span>
                            </label>
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
                <ModelEvaluationSection data={evaluationData} />
            )}
        </div>
    );
};

export default EvaluatePage;
