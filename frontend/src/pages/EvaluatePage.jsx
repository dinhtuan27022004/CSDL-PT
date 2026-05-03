import React, { useEffect, useState } from 'react';
import { 
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
    Cell, LabelList, LineChart, Line
} from 'recharts';
import { 
    GitCompare, 
    Play, 
    RefreshCw, 
    TrendingUp, 
    Target, 
    Layers,
    Info,
    AlertCircle,
    CheckCircle2,
    Zap
} from 'lucide-react';
import useImageStore from '../store/useImageStore';

const EvaluatePage = () => {
    const { 
        evaluationAllData, 
        evaluationLoading, 
        evaluationError, 
        fetchAllEvaluations, 
        triggerOptimization,
        isOptimizing,
        setIsOptimizing
    } = useImageStore();

    const [trials, setTrials] = useState(50);
    const [allowNegative, setAllowNegative] = useState(false);
    const [activeTab, setActiveTab] = useState('clip');

    useEffect(() => {
        fetchAllEvaluations();
    }, [fetchAllEvaluations]);

    const handleOptimize = async () => {
        await triggerOptimization('all', trials, allowNegative, true);
    };

    // Polling mechanism for background optimization
    useEffect(() => {
        let interval;
        if (isOptimizing) {
            // Poll for updates every 5 seconds
            interval = setInterval(() => {
                fetchAllEvaluations();
            }, 5000);
        }
        return () => {
            if (interval) clearInterval(interval);
        };
    }, [isOptimizing, fetchAllEvaluations]);

    const API_BASE_URL = 'http://localhost:8000';

    const ModelEvaluationSection = ({ modelKey, data }) => {
        if (!data) return null;

        const chartData = data.weights ? 
            Object.entries(data.weights)
                .map(([name, value]) => ({ name, value: parseFloat((value * 100).toFixed(2)) }))
                .sort((a, b) => b.value - a.value)
                .filter(item => item.value > 0.5)
            : [];

        const prCurveUrl = `${API_BASE_URL}/static/visualizations/pr_curve_${modelKey}.png`;
        const pkCurveUrl = `${API_BASE_URL}/static/visualizations/pk_curve_${modelKey}.png`;
        const scoreDistUrl = `${API_BASE_URL}/static/visualizations/score_dist_${modelKey}.png`;

        return (
            <div className="space-y-8 animate-in fade-in duration-700">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Weights Distribution */}
                    <div className="bg-slate-800/40 p-8 rounded-2xl border border-slate-700/50 shadow-xl h-fit sticky top-8">
                        <h3 className="text-xl font-bold text-white mb-8 flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2">
                                <Layers className="w-5 h-5 text-primary-500" />
                                Optimized Weights (%)
                            </div>
                        </h3>
                        <div className="w-full" style={{ height: `${Math.max(500, chartData.length * 18)}px` }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 60, left: 80, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                                    <XAxis type="number" hide />
                                    <YAxis 
                                        dataKey="name" 
                                        type="category" 
                                        stroke="#94a3b8" 
                                        fontSize={9} 
                                        width={70}
                                        interval={0}
                                        tick={{ fill: '#94a3b8' }}
                                    />
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

                    {/* All Charts Vertical Stack */}
                    <div className="space-y-6">
                        {/* Summary Metrics */}
                        <div className="bg-primary-900/10 p-6 rounded-2xl border border-primary-500/20 flex items-center justify-around">
                            <div className="text-center">
                                <span className="text-primary-400 font-bold uppercase text-[10px] tracking-widest block mb-1">mAP Total</span>
                                <span className="text-2xl font-black text-white">{((data.metrics?.test_map_after || 0) * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-10 w-px bg-primary-500/20"></div>
                            <div className="text-center">
                                <span className="text-primary-400 font-bold uppercase text-[10px] tracking-widest block mb-1">mAP @ Top 5</span>
                                <span className="text-2xl font-black text-white">{((data.metrics?.test_map5_after || 0) * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-10 w-px bg-primary-500/20"></div>
                            <div className="text-center">
                                <span className="text-primary-400 font-bold uppercase text-[10px] tracking-widest block mb-1">mAP @ Top 10</span>
                                <span className="text-2xl font-black text-white">{((data.metrics?.test_map10_after || 0) * 100).toFixed(1)}%</span>
                            </div>
                        </div>

                        {/* PR Curve */}
                        <div className="bg-slate-800/40 p-6 rounded-2xl border border-slate-700/50">
                            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                                <TrendingUp className="w-3 h-3 text-primary-500" /> Precision-Recall Analysis
                            </h4>
                            <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/30">
                                <img src={`${prCurveUrl}?t=${data.timestamp}`} alt="PR Curve" className="w-full object-contain" />
                            </div>
                        </div>

                        {/* PK Curve */}
                        <div className="bg-slate-800/40 p-6 rounded-2xl border border-slate-700/50">
                            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                                <Target className="w-3 h-3 text-primary-500" /> Precision at K (1-20)
                            </h4>
                            <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/30">
                                <img src={`${pkCurveUrl}?t=${data.timestamp}`} alt="P@K Curve" className="w-full object-contain" />
                            </div>
                        </div>

                        {/* Separation */}
                        <div className="bg-slate-800/40 p-6 rounded-2xl border border-slate-700/50">
                            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                                <Zap className="w-3 h-3 text-primary-500" /> Score Separation (GT vs Others)
                            </h4>
                            <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/30">
                                <img src={`${scoreDistUrl}?t=${data.timestamp}`} alt="Score Separation" className="w-full object-contain" />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    if (evaluationLoading && Object.keys(evaluationAllData).length === 0) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[60vh] text-slate-400">
                <RefreshCw className="w-12 h-12 mb-4 animate-spin text-primary-500" />
                <p className="text-lg animate-pulse">Analyzing multi-model performance...</p>
            </div>
        );
    }

    const availableModels = Object.keys(evaluationAllData);

    return (
        <div className="max-w-7xl mx-auto space-y-8 pb-12">
            {/* Unified Control Center */}
            <div className="bg-slate-800/40 backdrop-blur-md p-8 rounded-3xl border border-slate-700/50 shadow-2xl overflow-hidden relative">
                <div className="absolute top-0 right-0 w-64 h-64 bg-primary-500/5 rounded-full blur-3xl -mr-32 -mt-32"></div>
                
                <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-8">
                    <div className="flex-1">
                        <h1 className="text-4xl font-black text-white flex items-center gap-4">
                            <GitCompare className="w-10 h-10 text-primary-500" />
                            Optimization
                        </h1>
                     
                    </div>

                    <div className="flex flex-wrap items-center gap-6 bg-slate-900/60 p-6 rounded-2xl border border-slate-700/30">
                        <div className="space-y-1.5">
                            
                            <input 
                                type="number" value={trials} onChange={(e) => setTrials(parseInt(e.target.value))}
                                className="bg-slate-800 border border-slate-700 text-white text-sm rounded-xl focus:ring-primary-500 block w-24 p-2.5"
                            />
                        </div>
                        <div className="flex items-center gap-3">
                            <label htmlFor="neg" className="relative inline-flex items-center cursor-pointer group">
                                <input 
                                    type="checkbox" 
                                    id="neg" 
                                    checked={allowNegative} 
                                    onChange={(e) => setAllowNegative(e.target.checked)} 
                                    className="sr-only peer" 
                                />
                                <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600 group-hover:ring-4 group-hover:ring-primary-500/10 transition-all"></div>
                                <span className="ml-3 text-sm font-bold text-slate-400 group-hover:text-slate-300 transition-colors">Negative Weights</span>
                            </label>
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={handleOptimize} disabled={isOptimizing}
                                className={`flex items-center gap-3 px-8 py-3 rounded-xl font-black transition-all duration-300 ${
                                    isOptimizing ? 'bg-primary-600/20 text-primary-400 border border-primary-500/30 cursor-default' : 'bg-primary-600 hover:bg-primary-500 text-white shadow-xl shadow-primary-600/30 hover:-translate-y-1'
                                }`}
                            >
                                {isOptimizing ? <RefreshCw className="w-6 h-6 animate-spin" /> : <Play className="w-6 h-6 fill-current" />}
                                {isOptimizing ? 'OPTIMIZING (LIVE UPDATING...)' : 'RUN ALL OPTIMIZATIONS'}
                            </button>
                            {isOptimizing && (
                                <button 
                                    onClick={() => setIsOptimizing(false)}
                                    className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/30 rounded-xl text-xs font-bold transition-all"
                                >
                                    STOP POLLING
                                </button>
                            )}
                            <button onClick={fetchAllEvaluations} className="p-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 border border-slate-700 transition-all">
                                <RefreshCw className="w-6 h-6" />
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {availableModels.length === 0 ? (
                <div className="bg-slate-800/20 border-2 border-dashed border-slate-700/50 p-20 rounded-3xl text-center">
                    <AlertCircle className="w-16 h-16 text-slate-600 mx-auto mb-6" />
                    <h3 className="text-2xl font-bold text-slate-400">No Multi-Model Results Yet</h3>
                    <p className="text-slate-500 mt-2 mb-8">Run the optimization process to generate performance data for all ground truth models.</p>
                    <button onClick={handleOptimize} className="px-8 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl font-bold transition-all border border-slate-700">
                        Start First Run
                    </button>
                </div>
            ) : (
                <div className="space-y-10">
                    {/* Tab Navigation */}
                    <div className="flex p-1.5 bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-700/50 w-fit mx-auto lg:mx-0">
                        {['clip', 'dinov2', 'siglip', 'dreamsim'].map((key) => (
                            <button
                                key={key} onClick={() => setActiveTab(key)}
                                className={`px-8 py-3 rounded-xl text-sm font-black transition-all uppercase tracking-widest ${
                                    activeTab === key ? 'bg-primary-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'
                                }`}
                            >
                                {key} {evaluationAllData[key] ? '✓' : ''}
                            </button>
                        ))}
                    </div>

                    {/* Active Model Result */}
                    {evaluationAllData[activeTab] ? (
                        <ModelEvaluationSection 
                            key={activeTab}
                            modelKey={activeTab} 
                            data={evaluationAllData[activeTab]} 
                        />
                    ) : (
                        <div className="p-12 text-center bg-slate-800/20 rounded-3xl border border-slate-700/30">
                            <p className="text-slate-500 italic uppercase tracking-widest">No data available for {activeTab.toUpperCase()}</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default EvaluatePage;
