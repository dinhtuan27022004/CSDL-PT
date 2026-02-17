import React, { useState, useRef } from 'react';
import useImageStore from '../../store/useImageStore';
import Card from '../common/Card';
import Button from '../common/Button';
import { Upload, FileImage, X, Image as ImageIcon, Search } from 'lucide-react';

const SimilaritySearchForm = () => {
    const { searchImages, similarityLoading } = useImageStore();
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [limit, setLimit] = useState(5);
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
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = (newFile) => {
        if (newFile.type.startsWith('image/')) {
            setFile(newFile);
            setPreviewUrl(URL.createObjectURL(newFile));
        }
    };

    const removeFile = () => {
        setFile(null);
        setPreviewUrl(null);
        if (inputRef.current) {
            inputRef.current.value = '';
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) return;
        await searchImages(file, limit);
    };

    const onButtonClick = () => {
        inputRef.current.click();
    };

    return (
        <Card className="p-6">
            <div className="flex items-center gap-2 mb-4">
                <Search className="w-5 h-5 text-primary-400" />
                <h3 className="text-xl font-semibold text-white">Similarity Search</h3>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4" onDragEnter={handleDrag}>
                {/* Drag and Drop Zone - Only show if no file selected */}
                {!file ? (
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
                            accept="image/*"
                            onChange={handleChange}
                            className="hidden"
                        />

                        <FileImage className="w-12 h-12 mx-auto text-slate-400 mb-4" />

                        <p className="text-lg text-slate-200 font-medium mb-1">
                            Drag & Drop search image here
                        </p>
                        <p className="text-sm text-slate-500 mb-4">
                            or click to browse
                        </p>

                        <Button
                            type="button"
                            variant="secondary"
                            onClick={onButtonClick}
                            disabled={similarityLoading}
                        >
                            Select Image
                        </Button>
                    </div>
                ) : (
                    // Selected File Preview
                    <div className="bg-slate-800/50 rounded-lg p-4 flex items-center justify-between border border-slate-700">
                        <div className="flex items-center gap-4">
                            <div className="w-16 h-16 rounded bg-slate-700 overflow-hidden shrink-0">
                                <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
                            </div>
                            <div>
                                <h4 className="text-sm font-medium text-slate-200">{file.name}</h4>
                                <p className="text-xs text-slate-500">{(file.size / 1024).toFixed(1)} KB</p>
                            </div>
                        </div>
                        <button
                            type="button"
                            onClick={removeFile}
                            className="p-2 hover:bg-slate-700 rounded-full text-slate-400 hover:text-white transition-colors"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                )}

                {/* Search Options */}
                <div className="flex items-center gap-4">
                    <div className="flex-1">
                        <label className="block text-sm font-medium text-slate-400 mb-1">
                            Number of Results
                        </label>
                        <input
                            type="number"
                            min="1"
                            max="50"
                            value={limit}
                            onChange={(e) => setLimit(parseInt(e.target.value) || 5)}
                            className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white focus:outline-none focus:border-primary-500 transition-colors"
                        />
                    </div>
                    <div className="flex-none pt-6">
                        <Button
                            type="submit"
                            variant="primary"
                            className="min-w-[120px]"
                            disabled={similarityLoading || !file}
                        >
                            {similarityLoading ? 'Searching...' : 'Search'}
                        </Button>
                    </div>
                </div>
            </form>
        </Card>
    );
};

export default SimilaritySearchForm;
