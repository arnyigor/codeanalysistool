package com.example.dataprocessing

data class AnalysisResult(
    val isValid: Boolean,
    val data: String?,
    val processingTime: Long,
    val error: String?
) 