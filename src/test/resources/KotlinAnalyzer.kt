package com.example

import java.time.LocalDateTime
import kotlin.random.Random
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.ConcurrentHashMap

@Suppress("TooManyFunctions")
class KotlinAnalyzer<T : Any> {
    private val patterns = ConcurrentHashMap<String, Pattern>()
    private val processingTimes = ConcurrentHashMap<String, ProcessingMetrics>()
    private val analysisCache = ConcurrentHashMap<String, AnalysisResult<T>>()
    
    data class Pattern(
        val regex: Regex,
        val priority: Int,
        val description: String
    )
    
    data class ProcessingMetrics(
        val startTime: LocalDateTime,
        val duration: Long,
        val memoryUsage: Long,
        val threadId: String = Thread.currentThread().name
    )
    
    sealed class AnalysisError {
        data class PatternNotFound(val input: String) : AnalysisError()
        data class ProcessingError(val message: String, val cause: Throwable? = null) : AnalysisError()
        data class ValidationError(val violations: List<String>) : AnalysisError()
    }
    
    data class AnalysisResult<T>(
        val isValid: Boolean,
        val data: T?,
        val processingTime: Long,
        val error: AnalysisError? = null,
        val metadata: Map<String, Any> = emptyMap()
    )
    
    @Throws(IllegalStateException::class)
    suspend fun analyzeData(input: String): AnalysisResult<T> = withContext(Dispatchers.Default) {
        val startTime = System.currentTimeMillis()
        val startMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
        
        try {
            // Проверка кэша
            analysisCache[input]?.let { return@withContext it }
            
            // Валидация входных данных
            validateInput(input).let {
                if (it.isNotEmpty()) {
                    return@withContext AnalysisResult(
                        isValid = false,
                        data = null,
                        processingTime = 0,
                        error = AnalysisError.ValidationError(it)
                    )
                }
            }
            
            // Поиск подходящего паттерна
            val matchedPattern = findBestPattern(input) ?: return@withContext AnalysisResult(
                isValid = false,
                data = null,
                processingTime = 0,
                error = AnalysisError.PatternNotFound(input)
            )
            
            // Обработка данных
            val processedData = processData(input, matchedPattern)
            val endTime = System.currentTimeMillis()
            val endMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            
            // Сохранение метрик
            val metrics = ProcessingMetrics(
                startTime = LocalDateTime.now(),
                duration = endTime - startTime,
                memoryUsage = endMemory - startMemory
            )
            processingTimes[input] = metrics
            
            // Формирование результата
            val result = AnalysisResult(
                isValid = true,
                data = processedData as T,
                processingTime = metrics.duration,
                metadata = mapOf(
                    "pattern" to matchedPattern.description,
                    "memoryUsage" to metrics.memoryUsage,
                    "threadId" to metrics.threadId
                )
            )
            
            // Кэширование результата
            analysisCache[input] = result
            return@withContext result
            
        } catch (e: Exception) {
            return@withContext AnalysisResult(
                isValid = false,
                data = null,
                processingTime = System.currentTimeMillis() - startTime,
                error = AnalysisError.ProcessingError(e.message ?: "Unknown error", e)
            )
        }
    }
    
    private fun validateInput(input: String): List<String> {
        val violations = mutableListOf<String>()
        if (input.isBlank()) violations.add("Input cannot be blank")
        if (input.length < 3) violations.add("Input must be at least 3 characters long")
        if (input.length > 1000) violations.add("Input must not exceed 1000 characters")
        return violations
    }
    
    private fun findBestPattern(input: String): Pattern? {
        return patterns.values
            .filter { it.regex.matches(input) }
            .maxByOrNull { it.priority }
    }
    
    @Suppress("UNCHECKED_CAST")
    private fun processData(input: String, pattern: Pattern): Any {
        return buildString {
            append("[${LocalDateTime.now()}] ")
            append("Pattern '${pattern.description}' (priority: ${pattern.priority}) matched: ")
            append(input)
        }
    }
    
    fun addPattern(regex: String, priority: Int = 0, description: String) {
        patterns[regex] = Pattern(regex.toRegex(), priority, description)
    }
    
    fun removePattern(regex: String) {
        patterns.remove(regex)
    }
    
    fun getPatterns(): Map<String, Pattern> = patterns.toMap()
    
    fun getProcessingMetrics(): Flow<ProcessingMetrics> = flow {
        processingTimes.values.forEach { emit(it) }
    }
    
    fun getAverageProcessingTime(): Double {
        return processingTimes.values
            .map { it.duration }
            .average()
    }
    
    fun getAverageMemoryUsage(): Double {
        return processingTimes.values
            .map { it.memoryUsage }
            .average()
    }
    
    fun clearMetrics() {
        processingTimes.clear()
    }
    
    fun clearCache() {
        analysisCache.clear()
    }
    
    fun clearAll() {
        patterns.clear()
        clearMetrics()
        clearCache()
    }
} 