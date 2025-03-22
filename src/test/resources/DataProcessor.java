package com.example.dataprocessing;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.time.LocalDateTime;
import java.time.Duration;
import com.example.KotlinAnalyzer;

@SuppressWarnings("unchecked")
public class DataProcessor<T> {
    private final KotlinAnalyzer<T> analyzer;
    private final List<T> processedData;
    private final Map<String, ProcessingMetrics> metricsHistory;
    private final Map<String, Function<T, Boolean>> validators;
    
    public static class ProcessingMetrics {
        private final LocalDateTime startTime;
        private final Duration duration;
        private final long memoryUsage;
        private final String threadId;
        
        public ProcessingMetrics(LocalDateTime startTime, Duration duration, long memoryUsage, String threadId) {
            this.startTime = startTime;
            this.duration = duration;
            this.memoryUsage = memoryUsage;
            this.threadId = threadId;
        }
        
        public LocalDateTime getStartTime() { return startTime; }
        public Duration getDuration() { return duration; }
        public long getMemoryUsage() { return memoryUsage; }
        public String getThreadId() { return threadId; }
    }
    
    public static class ProcessingException extends RuntimeException {
        private final ErrorType errorType;
        private final Map<String, Object> details;
        
        public enum ErrorType {
            VALIDATION_ERROR,
            PROCESSING_ERROR,
            SYSTEM_ERROR
        }
        
        public ProcessingException(ErrorType errorType, String message, Map<String, Object> details) {
            super(message);
            this.errorType = errorType;
            this.details = details;
        }
        
        public ErrorType getErrorType() { return errorType; }
        public Map<String, Object> getDetails() { return details; }
    }
    
    public DataProcessor(KotlinAnalyzer<T> analyzer) {
        this.analyzer = analyzer;
        this.processedData = new ArrayList<>();
        this.metricsHistory = new ConcurrentHashMap<>();
        this.validators = new ConcurrentHashMap<>();
    }
    
    public CompletableFuture<T> processDataAsync(String input) {
        return CompletableFuture.supplyAsync(() -> {
            LocalDateTime startTime = LocalDateTime.now();
            long startMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            
            try {
                // Валидация входных данных
                validateInput(input);
                
                // Асинхронная обработка через Kotlin analyzer
                T result = analyzer.analyzeData(input).getData();
                
                // Валидация результата
                if (!validateResult(result)) {
                    throw new ProcessingException(
                        ProcessingException.ErrorType.VALIDATION_ERROR,
                        "Result validation failed",
                        Map.of("input", input, "result", result)
                    );
                }
                
                // Сохранение результата
                processedData.add(result);
                
                // Сохранение метрик
                LocalDateTime endTime = LocalDateTime.now();
                long endMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                
                metricsHistory.put(input, new ProcessingMetrics(
                    startTime,
                    Duration.between(startTime, endTime),
                    endMemory - startMemory,
                    Thread.currentThread().getName()
                ));
                
                return result;
                
            } catch (Exception e) {
                throw new ProcessingException(
                    ProcessingException.ErrorType.PROCESSING_ERROR,
                    "Processing failed: " + e.getMessage(),
                    Map.of("input", input, "error", e.getMessage())
                );
            }
        });
    }
    
    private void validateInput(String input) {
        if (input == null || input.trim().isEmpty()) {
            throw new ProcessingException(
                ProcessingException.ErrorType.VALIDATION_ERROR,
                "Input cannot be null or empty",
                Map.of("input", input)
            );
        }
    }
    
    private boolean validateResult(T result) {
        return validators.values().stream()
            .allMatch(validator -> validator.apply(result));
    }
    
    public void addValidator(String name, Function<T, Boolean> validator) {
        validators.put(name, validator);
    }
    
    public void removeValidator(String name) {
        validators.remove(name);
    }
    
    public List<T> getProcessedData() {
        return new ArrayList<>(processedData);
    }
    
    public Map<String, ProcessingMetrics> getMetricsHistory() {
        return new ConcurrentHashMap<>(metricsHistory);
    }
    
    public double getAverageProcessingTime() {
        return metricsHistory.values().stream()
            .mapToLong(m -> m.getDuration().toMillis())
            .average()
            .orElse(0.0);
    }
    
    public double getAverageMemoryUsage() {
        return metricsHistory.values().stream()
            .mapToLong(ProcessingMetrics::getMemoryUsage)
            .average()
            .orElse(0.0);
    }
    
    public void clearProcessedData() {
        processedData.clear();
    }
    
    public void clearMetrics() {
        metricsHistory.clear();
    }
    
    public void clearAll() {
        clearProcessedData();
        clearMetrics();
        validators.clear();
    }
} 