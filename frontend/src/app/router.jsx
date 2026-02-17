import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from '../components/layout/MainLayout';
import ImportPage from '../pages/ImportPage';
import SimilaritySearchPage from '../pages/SimilaritySearchPage';

const AppRouter = () => {
    return (
        <Routes>
            <Route path="/" element={<MainLayout />}>
                <Route index element={<Navigate to="/import" replace />} />
                <Route path="import" element={<ImportPage />} />
                <Route path="search" element={<SimilaritySearchPage />} />
            </Route>
        </Routes>
    );
};

export default AppRouter;
