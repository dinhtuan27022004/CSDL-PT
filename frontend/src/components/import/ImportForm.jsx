import React, { useState, useRef } from 'react';
import useImageStore from '../../store/useImageStore';
import Card from '../common/Card';
import Button from '../common/Button';
import ProgressBar from '../common/ProgressBar'; // Assuming this component exists
import { Upload, FileImage, X, Image as ImageIcon } from 'lucide-react';

const ImportForm = () => {
    const { addImage, importLoading, importProgress } = useImageStore();
    const [dragActive, setDragActive] = useState(false);
    const [files, setFiles] = useState([]);
    const inputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files.length > 0) {
            handleFiles(e.target.files);
        }
    };

    const handleFiles = (newFiles) => {
        const validFiles = Array.from(newFiles).filter(file => file.type.startsWith('image/'));
        if (validFiles.length > 0) {
            setFiles(prev => [...prev, ...validFiles]);
        }
    };

    const removeFile = (index) => {
        setFiles(prev => prev.filter((_, i) => i !== index));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (files.length === 0) return;

        try {
            await addImage(files);
            setFiles([]); // Clear after successful import
        } catch (error) {
            console.error('Import failed:', error);
        }
    };

    const onButtonClick = () => {
        inputRef.current.click();
    };

    return (
        <Card className="p-6">
            <div className="flex items-center gap-2 mb-4">
                <Upload className="w-5 h-5 text-primary-400" />
                <h3 className="text-xl font-semibold text-white">Import Images</h3>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4" onDragEnter={handleDrag}>
                {/* Drag and Drop Zone */}
                <div
                    className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${dragActive
                            ? "border-primary-500 bg-primary-500/10"
                            : "border-slate-600 hover:border-slate-500 bg-slate-800/30"
                        }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                >
                    <input
                        ref={inputRef}
                        type="file"
                        multiple
                        webkitdirectory=""
                        directory=""
                        accept="image/*"
                        onChange={handleChange}
                        className="hidden"
                    />

                    <FileImage className="w-12 h-12 mx-auto text-slate-400 mb-4" />

                    <p className="text-lg text-slate-200 font-medium mb-1">
                        Drag & Drop images here
                    </p>
                    <p className="text-sm text-slate-500 mb-4">
                        or click to browse files
                    </p>

                    <Button
                        type="button"
                        variant="secondary"
                        onClick={onButtonClick}
                        disabled={importLoading}
                    >
                        Select Files
                    </Button>
                </div>

                {/* File Preview List */}
                {files.length > 0 && (
                    <div className="bg-slate-800/50 rounded-lg p-4 space-y-2 max-h-60 overflow-y-auto">
                        <div className="flex justify-between items-center text-sm text-slate-400 mb-2">
                            <span>Selected Files ({files.length})</span>
                            <button
                                type="button"
                                onClick={() => setFiles([])}
                                className="text-red-400 hover:text-red-300"
                            >
                                Clear All
                            </button>
                        </div>
                        {files.map((file, index) => (
                            <div key={`${file.name}-${index}`} className="flex items-center justify-between bg-slate-700/50 p-2 rounded">
                                <div className="flex items-center gap-3 overflow-hidden">
                                    <div className="w-8 h-8 rounded bg-slate-600 flex items-center justify-center shrink-0">
                                        <ImageIcon className="w-4 h-4 text-slate-400" />
                                    </div>
                                    <span className="text-sm text-slate-200 truncate">{file.name}</span>
                                    <span className="text-xs text-slate-500">
                                        {(file.size / 1024).toFixed(1)} KB
                                    </span>
                                </div>
                                <button
                                    type="button"
                                    onClick={() => removeFile(index)}
                                    className="p-1 hover:bg-slate-600 rounded text-slate-400 hover:text-white transition-colors"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                {/* Progress Bar */}
                {importLoading && (
                    <div>
                        <div className="flex justify-between text-sm text-slate-400 mb-2">
                            <span>Importing...</span>
                            <span>{Math.round(importProgress)}%</span>
                        </div>
                        <ProgressBar progress={importProgress} />
                    </div>
                )}

                {/* Submit Action */}
                <div className="flex justify-end pt-2">
                    <Button
                        type="submit"
                        variant="primary"
                        className="w-full sm:w-auto"
                        disabled={importLoading || files.length === 0}
                    >
                        {importLoading ? 'Processing...' : `Import ${files.length} Images`}
                    </Button>
                </div>
            </form>
        </Card>
    );
};

export default ImportForm;
