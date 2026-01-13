"""
Incremental training script for CODE DATASETS.

Trains CRAYON vocabulary on comprehensive programming language patterns.
Uses built-in code samples from multiple languages + optional HuggingFace datasets.

Objective:
- Load existing 'trained_vocab.json'.
- Train on comprehensive code samples (Python, JS, Java, C++, Rust, Go, etc.).
- Optionally stream from HuggingFace if available.
- Merge NEW tokens into existing vocabulary (append-only, ID-stable).
"""

import json
import time
import logging
import sys
from pathlib import Path
from typing import Iterator, Set, List, Optional
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from crayon import CrayonVocab
from crayon.training import train_vocabulary

# ============================================================================
# Configuration
# ============================================================================

EXISTING_VOCAB_PATH = Path("trained_vocab.json")

# ============================================================================
# COMPREHENSIVE CODE SAMPLES - Multiple Languages
# ============================================================================

PYTHON_SAMPLES = [
    # Functions and classes
    '''
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n: int) -> int:
    """Calculate factorial using iteration."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

class DataProcessor:
    """Process data with various transformations."""
    
    def __init__(self, data: list, config: dict = None):
        self.data = data
        self.config = config or {}
        self._cache = {}
    
    def process(self) -> list:
        """Apply transformations to data."""
        return [self._transform(x) for x in self.data if self._validate(x)]
    
    def _transform(self, item):
        return item * 2 if isinstance(item, (int, float)) else str(item)
    
    def _validate(self, item) -> bool:
        return item is not None

    @property
    def processed_count(self) -> int:
        return len(self._cache)
    
    @staticmethod
    def from_file(path: str) -> 'DataProcessor':
        with open(path, 'r') as f:
            data = json.load(f)
        return DataProcessor(data)

    @classmethod
    def create_empty(cls) -> 'DataProcessor':
        return cls([])
''',
    # Async/await patterns
    '''
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional

async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    """Fetch data from URL asynchronously."""
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        raise ValueError(f"HTTP {response.status}: {url}")

async def fetch_all(urls: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

async def process_stream(reader: asyncio.StreamReader) -> bytes:
    """Process a stream of data."""
    chunks = []
    async for chunk in reader:
        chunks.append(chunk)
    return b''.join(chunks)
''',
    # Data science patterns
    '''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def train_model(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Pandas operations
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df["c"] = df["a"] + df["b"]
df = df.groupby("a").agg({"b": "sum", "c": "mean"})
df = df.merge(other_df, on="key", how="left")
df.to_csv("output.csv", index=False)
''',
    # Context managers and decorators
    '''
from functools import wraps
from contextlib import contextmanager
import threading
import time

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (attempt + 1))
        return wrapper
    return decorator

@contextmanager
def database_connection(connection_string: str):
    conn = create_connection(connection_string)
    try:
        yield conn
    finally:
        conn.close()

class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self) -> int:
        with self._lock:
            return self._value
''',
    # Type hints and protocols
    '''
from typing import (
    List, Dict, Set, Tuple, Optional, Union, Any, Callable,
    TypeVar, Generic, Protocol, runtime_checkable, Literal,
    Awaitable, Iterable, Iterator, Generator
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other: Any) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...

@dataclass
class Config:
    name: str
    value: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class Status(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

class Repository(ABC, Generic[T]):
    @abstractmethod
    def get(self, id: str) -> Optional[T]: ...
    
    @abstractmethod
    def save(self, item: T) -> None: ...
    
    @abstractmethod
    def delete(self, id: str) -> bool: ...

def process_items(
    items: Iterable[T],
    transform: Callable[[T], V],
    filter_fn: Optional[Callable[[T], bool]] = None
) -> Generator[V, None, None]:
    for item in items:
        if filter_fn is None or filter_fn(item):
            yield transform(item)
''',
    # Exception handling
    '''
class ValidationError(Exception):
    """Raised when validation fails."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

class APIError(Exception):
    """Base class for API errors."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")

class NotFoundError(APIError):
    def __init__(self, resource: str):
        super().__init__(404, f"{resource} not found")

def safe_divide(a: float, b: float) -> Optional[float]:
    try:
        return a / b
    except ZeroDivisionError:
        logger.warning("Division by zero attempted")
        return None
    except TypeError as e:
        logger.error(f"Type error: {e}")
        raise ValueError(f"Invalid types: {type(a)}, {type(b)}") from e
    finally:
        logger.debug("Division operation completed")
''',
]

JAVASCRIPT_SAMPLES = [
    # Modern JS patterns
    '''
// Arrow functions and destructuring
const processData = ({ id, name, value = 0 }) => ({
    id,
    displayName: name.toUpperCase(),
    processedValue: value * 2,
    timestamp: Date.now()
});

const fetchData = async (url, options = {}) => {
    try {
        const response = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Fetch failed:', error);
        throw error;
    }
};

// Promise patterns
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const retryWithBackoff = async (fn, maxRetries = 3) => {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            await delay(Math.pow(2, i) * 1000);
        }
    }
};

// Array methods
const users = [
    { id: 1, name: 'Alice', active: true },
    { id: 2, name: 'Bob', active: false },
    { id: 3, name: 'Charlie', active: true }
];

const activeUserNames = users
    .filter(user => user.active)
    .map(user => user.name)
    .sort((a, b) => a.localeCompare(b));

const userById = users.reduce((acc, user) => {
    acc[user.id] = user;
    return acc;
}, {});
''',
    # Classes and modules
    '''
// ES6+ Class syntax
class EventEmitter {
    #listeners = new Map();
    
    on(event, callback) {
        if (!this.#listeners.has(event)) {
            this.#listeners.set(event, new Set());
        }
        this.#listeners.get(event).add(callback);
        return () => this.off(event, callback);
    }
    
    off(event, callback) {
        this.#listeners.get(event)?.delete(callback);
    }
    
    emit(event, ...args) {
        this.#listeners.get(event)?.forEach(cb => cb(...args));
    }
    
    once(event, callback) {
        const wrapper = (...args) => {
            callback(...args);
            this.off(event, wrapper);
        };
        return this.on(event, wrapper);
    }
}

class AsyncQueue {
    #queue = [];
    #processing = false;
    
    async add(task) {
        return new Promise((resolve, reject) => {
            this.#queue.push({ task, resolve, reject });
            this.#process();
        });
    }
    
    async #process() {
        if (this.#processing) return;
        this.#processing = true;
        
        while (this.#queue.length > 0) {
            const { task, resolve, reject } = this.#queue.shift();
            try {
                resolve(await task());
            } catch (error) {
                reject(error);
            }
        }
        
        this.#processing = false;
    }
}

export { EventEmitter, AsyncQueue };
export default EventEmitter;
''',
    # React patterns
    '''
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';

const useDebounce = (value, delay) => {
    const [debouncedValue, setDebouncedValue] = useState(value);
    
    useEffect(() => {
        const timer = setTimeout(() => setDebouncedValue(value), delay);
        return () => clearTimeout(timer);
    }, [value, delay]);
    
    return debouncedValue;
};

const useFetch = (url) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    useEffect(() => {
        const controller = new AbortController();
        
        const fetchData = async () => {
            try {
                setLoading(true);
                const response = await fetch(url, { signal: controller.signal });
                const json = await response.json();
                setData(json);
            } catch (err) {
                if (err.name !== 'AbortError') {
                    setError(err);
                }
            } finally {
                setLoading(false);
            }
        };
        
        fetchData();
        return () => controller.abort();
    }, [url]);
    
    return { data, loading, error };
};

const SearchComponent = ({ onSearch }) => {
    const [query, setQuery] = useState('');
    const debouncedQuery = useDebounce(query, 300);
    const inputRef = useRef(null);
    
    useEffect(() => {
        if (debouncedQuery) {
            onSearch(debouncedQuery);
        }
    }, [debouncedQuery, onSearch]);
    
    const handleChange = useCallback((e) => {
        setQuery(e.target.value);
    }, []);
    
    return (
        <div className="search-container">
            <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={handleChange}
                placeholder="Search..."
            />
        </div>
    );
};

export default SearchComponent;
''',
]

TYPESCRIPT_SAMPLES = [
    '''
// TypeScript interfaces and types
interface User {
    id: number;
    name: string;
    email: string;
    role: 'admin' | 'user' | 'guest';
    createdAt: Date;
    metadata?: Record<string, unknown>;
}

type PartialUser = Partial<User>;
type RequiredUser = Required<User>;
type UserKeys = keyof User;
type ReadonlyUser = Readonly<User>;

interface Repository<T> {
    find(id: string): Promise<T | null>;
    findAll(): Promise<T[]>;
    create(item: Omit<T, 'id'>): Promise<T>;
    update(id: string, item: Partial<T>): Promise<T>;
    delete(id: string): Promise<boolean>;
}

// Generic constraints
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
    return obj[key];
}

// Conditional types
type NonNullable<T> = T extends null | undefined ? never : T;
type ExtractArrayType<T> = T extends Array<infer U> ? U : never;

// Utility implementations
class UserRepository implements Repository<User> {
    private users: Map<string, User> = new Map();
    
    async find(id: string): Promise<User | null> {
        return this.users.get(id) ?? null;
    }
    
    async findAll(): Promise<User[]> {
        return Array.from(this.users.values());
    }
    
    async create(item: Omit<User, 'id'>): Promise<User> {
        const id = crypto.randomUUID();
        const user: User = { ...item, id: parseInt(id) };
        this.users.set(id, user);
        return user;
    }
    
    async update(id: string, item: Partial<User>): Promise<User> {
        const existing = await this.find(id);
        if (!existing) throw new Error('User not found');
        const updated = { ...existing, ...item };
        this.users.set(id, updated);
        return updated;
    }
    
    async delete(id: string): Promise<boolean> {
        return this.users.delete(id);
    }
}

// Decorators
function log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const original = descriptor.value;
    descriptor.value = function(...args: any[]) {
        console.log(`Calling ${propertyKey} with args:`, args);
        const result = original.apply(this, args);
        console.log(`${propertyKey} returned:`, result);
        return result;
    };
    return descriptor;
}
''']

JAVA_SAMPLES = [
    '''
package com.example.application;

import java.util.*;
import java.util.stream.*;
import java.util.concurrent.*;
import java.util.function.*;

public class DataProcessor<T extends Comparable<T>> {
    private final List<T> data;
    private final Map<String, Consumer<T>> handlers;
    
    public DataProcessor(List<T> data) {
        this.data = new ArrayList<>(data);
        this.handlers = new HashMap<>();
    }
    
    public List<T> process(Predicate<T> filter, Function<T, T> transform) {
        return data.stream()
            .filter(filter)
            .map(transform)
            .sorted()
            .collect(Collectors.toList());
    }
    
    public Map<Boolean, List<T>> partition(Predicate<T> predicate) {
        return data.stream()
            .collect(Collectors.partitioningBy(predicate));
    }
    
    public <R> R reduce(R identity, BiFunction<R, T, R> accumulator) {
        R result = identity;
        for (T item : data) {
            result = accumulator.apply(result, item);
        }
        return result;
    }
    
    public CompletableFuture<List<T>> processAsync(Executor executor) {
        return CompletableFuture.supplyAsync(() -> {
            return data.stream()
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
        }, executor);
    }
    
    @Override
    public String toString() {
        return String.format("DataProcessor{size=%d}", data.size());
    }
    
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        DataProcessor<Integer> processor = new DataProcessor<>(numbers);
        
        List<Integer> result = processor.process(
            n -> n % 2 == 0,
            n -> n * 2
        );
        
        System.out.println("Result: " + result);
    }
}

interface Repository<T, ID> {
    Optional<T> findById(ID id);
    List<T> findAll();
    T save(T entity);
    void delete(T entity);
    boolean existsById(ID id);
}

@FunctionalInterface
interface Validator<T> {
    boolean validate(T value);
    
    default Validator<T> and(Validator<T> other) {
        return value -> this.validate(value) && other.validate(value);
    }
}
''']

CPP_SAMPLES = [
    '''
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <functional>
#include <optional>
#include <variant>
#include <string_view>
#include <unordered_map>

template <typename T>
class SmartVector {
private:
    std::vector<T> data_;
    mutable std::optional<T> cached_sum_;
    
public:
    SmartVector() = default;
    explicit SmartVector(std::initializer_list<T> init) : data_(init) {}
    
    void push_back(T value) {
        data_.push_back(std::move(value));
        cached_sum_.reset();
    }
    
    template <typename... Args>
    void emplace_back(Args&&... args) {
        data_.emplace_back(std::forward<Args>(args)...);
        cached_sum_.reset();
    }
    
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }
    
    T& operator[](std::size_t index) { return data_[index]; }
    const T& operator[](std::size_t index) const { return data_[index]; }
    
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.cbegin(); }
    auto end() const { return data_.cend(); }
    
    template <typename Pred>
    [[nodiscard]] SmartVector filter(Pred predicate) const {
        SmartVector result;
        std::copy_if(data_.begin(), data_.end(),
                     std::back_inserter(result.data_), predicate);
        return result;
    }
    
    template <typename Func>
    [[nodiscard]] auto map(Func transform) const {
        using ResultType = std::invoke_result_t<Func, T>;
        SmartVector<ResultType> result;
        std::transform(data_.begin(), data_.end(),
                       std::back_inserter(result.data_), transform);
        return result;
    }
};

class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(std::string_view message) = 0;
};

class Subject {
    std::vector<std::weak_ptr<Observer>> observers_;
    
public:
    void attach(std::shared_ptr<Observer> observer) {
        observers_.push_back(observer);
    }
    
    void notify(std::string_view message) {
        observers_.erase(
            std::remove_if(observers_.begin(), observers_.end(),
                [&message](auto& weak) {
                    if (auto shared = weak.lock()) {
                        shared->update(message);
                        return false;
                    }
                    return true;
                }),
            observers_.end()
        );
    }
};

int main() {
    SmartVector<int> vec{1, 2, 3, 4, 5};
    
    auto filtered = vec.filter([](int x) { return x % 2 == 0; });
    auto mapped = filtered.map([](int x) { return x * x; });
    
    for (const auto& item : mapped) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
''']

RUST_SAMPLES = [
    '''
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct Config {
    pub name: String,
    pub value: i32,
    pub enabled: bool,
}

impl Config {
    pub fn new(name: impl Into<String>, value: i32) -> Self {
        Self {
            name: name.into(),
            value,
            enabled: true,
        }
    }
    
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct ConfigBuilder {
    name: Option<String>,
    value: Option<i32>,
    enabled: bool,
}

impl ConfigBuilder {
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    pub fn value(mut self, value: i32) -> Self {
        self.value = Some(value);
        self
    }
    
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
    
    pub fn build(self) -> Result<Config, &'static str> {
        Ok(Config {
            name: self.name.ok_or("name is required")?,
            value: self.value.unwrap_or(0),
            enabled: self.enabled,
        })
    }
}

pub trait Repository<T> {
    fn find(&self, id: &str) -> Option<&T>;
    fn find_all(&self) -> Vec<&T>;
    fn save(&mut self, id: String, item: T);
    fn delete(&mut self, id: &str) -> Option<T>;
}

pub struct InMemoryRepository<T> {
    data: HashMap<String, T>,
}

impl<T> InMemoryRepository<T> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl<T: Clone> Repository<T> for InMemoryRepository<T> {
    fn find(&self, id: &str) -> Option<&T> {
        self.data.get(id)
    }
    
    fn find_all(&self) -> Vec<&T> {
        self.data.values().collect()
    }
    
    fn save(&mut self, id: String, item: T) {
        self.data.insert(id, item);
    }
    
    fn delete(&mut self, id: &str) -> Option<T> {
        self.data.remove(id)
    }
}

async fn fetch_data(url: &str) -> Result<String, Box<dyn Error>> {
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::builder()
        .name("test")
        .value(42)
        .enabled(true)
        .build()?;
    
    println!("{:?}", config);
    
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Counter: {}", *counter.lock().unwrap());
    
    Ok(())
}
''']

GO_SAMPLES = [
    '''
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
    "time"
)

type User struct {
    ID        string    `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email"`
    CreatedAt time.Time `json:"created_at"`
}

type Repository[T any] interface {
    Find(ctx context.Context, id string) (*T, error)
    FindAll(ctx context.Context) ([]T, error)
    Save(ctx context.Context, item T) error
    Delete(ctx context.Context, id string) error
}

type InMemoryRepository[T any] struct {
    mu   sync.RWMutex
    data map[string]T
}

func NewInMemoryRepository[T any]() *InMemoryRepository[T] {
    return &InMemoryRepository[T]{
        data: make(map[string]T),
    }
}

func (r *InMemoryRepository[T]) Find(ctx context.Context, id string) (*T, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    
    item, ok := r.data[id]
    if !ok {
        return nil, fmt.Errorf("item not found: %s", id)
    }
    return &item, nil
}

func (r *InMemoryRepository[T]) FindAll(ctx context.Context) ([]T, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    
    items := make([]T, 0, len(r.data))
    for _, item := range r.data {
        items = append(items, item)
    }
    return items, nil
}

type Server struct {
    router *http.ServeMux
    repo   Repository[User]
}

func NewServer(repo Repository[User]) *Server {
    s := &Server{
        router: http.NewServeMux(),
        repo:   repo,
    }
    s.routes()
    return s
}

func (s *Server) routes() {
    s.router.HandleFunc("GET /users", s.handleGetUsers)
    s.router.HandleFunc("GET /users/{id}", s.handleGetUser)
    s.router.HandleFunc("POST /users", s.handleCreateUser)
}

func (s *Server) handleGetUsers(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    users, err := s.repo.FindAll(ctx)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(users)
}

func worker(ctx context.Context, jobs <-chan int, results chan<- int) {
    for {
        select {
        case <-ctx.Done():
            return
        case job, ok := <-jobs:
            if !ok {
                return
            }
            results <- job * 2
        }
    }
}

func main() {
    repo := NewInMemoryRepository[User]()
    server := NewServer(repo)
    
    fmt.Println("Starting server on :8080")
    http.ListenAndServe(":8080", server.router)
}
''']

# Common programming tokens to ensure coverage
PROGRAMMING_TOKENS = [
    # Python keywords
    "def ", "class ", "import ", "from ", "return ", "yield ", "async ", "await ",
    "if ", "elif ", "else:", "for ", "while ", "try:", "except ", "finally:",
    "with ", "as ", "lambda ", "pass", "break", "continue", "raise ", "assert ",
    "__init__", "__main__", "__name__", "__str__", "__repr__", "self.", "cls.",
    
    # JavaScript/TypeScript keywords  
    "function ", "const ", "let ", "var ", "export ", "import ", "async ",
    "await ", "=>", "===", "!==", "typeof ", "instanceof ", "Promise",
    "undefined", "null", ".then(", ".catch(", ".map(", ".filter(", ".reduce(",
    
    # Common operators and symbols
    "+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=",
    "==", "!=", "<=", ">=", "&&", "||", "++", "--", "<<", ">>",
    "->", "::", "...", "/**", "*/", "//", "/*", "#{", "${", "@",
    
    # Common patterns
    "print(", "console.log(", "System.out.", "printf(", "cout <<",
    ".append(", ".extend(", ".insert(", ".remove(", ".pop(",
    ".get(", ".set(", ".add(", ".update(", ".clear(",
    ".keys()", ".values()", ".items()", ".split(", ".join(",
    ".format(", ".replace(", ".strip(", ".lower()", ".upper()",
    
    # Type annotations
    ": int", ": str", ": float", ": bool", ": list", ": dict", ": set",
    ": List[", ": Dict[", ": Optional[", ": Tuple[", ": Union[",
    "-> None", "-> int", "-> str", "-> bool", "-> List",
    
    # Exception handling
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "ImportError", "OSError", "FileNotFoundError",
    
    # Java/C++ patterns
    "public ", "private ", "protected ", "static ", "final ", "void ",
    "String ", "Integer", "Boolean", "ArrayList", "HashMap", "System.",
    "#include", "#define", "namespace ", "template ", "std::",
    "nullptr", "virtual ", "override ", "const ", "struct ", "enum ",
    
    # Rust patterns
    "fn ", "let ", "mut ", "impl ", "pub ", "mod ", "use ", "crate ",
    "::new(", "unwrap(", "expect(", "Result<", "Option<",
    
    # Data science patterns
    "import numpy", "import pandas", "import torch", "import tensorflow",
    "np.", "pd.", "plt.", "torch.", "tf.", ".cuda()", ".numpy()",
    ".shape", ".dtype", ".fit(", ".predict(", ".transform(",
]


def yield_all_code_samples() -> Iterator[str]:
    """Yields all comprehensive code samples."""
    
    all_samples = (
        PYTHON_SAMPLES + 
        JAVASCRIPT_SAMPLES + 
        TYPESCRIPT_SAMPLES + 
        JAVA_SAMPLES + 
        CPP_SAMPLES + 
        RUST_SAMPLES + 
        GO_SAMPLES
    )
    
    print(f"[INFO] Loading {len(all_samples)} comprehensive code samples...")
    
    for sample in all_samples:
        yield sample
    
    # Also yield individual programming tokens
    for token in PROGRAMMING_TOKENS:
        yield token
    
    print(f"[INFO] Finished loading all code samples.")


def progress_callback(msg: str):
    """Progress callback that filters verbose output."""
    if "Processed" in msg and not msg.endswith("00 chunks..."):
        return
    print(f"[PROGRESS] {msg}")


def main():
    print("=" * 70)
    print("XERV Crayon: Incremental Training on Code Datasets")
    print("=" * 70)
    print()
    
    # 1. Load Existing Vocabulary
    print(f"[1] Loading existing vocabulary from {EXISTING_VOCAB_PATH}...")
    
    if not EXISTING_VOCAB_PATH.exists():
        print(f"    [ERROR] {EXISTING_VOCAB_PATH} not found!")
        print("    Run train_vocab.py first to create base vocabulary.")
        return
    
    try:
        base_vocab = CrayonVocab.from_json(str(EXISTING_VOCAB_PATH))
        base_size = len(base_vocab)
        print(f"    - Loaded {base_size:,} tokens")
        print(f"    - C-Extension: {'Enabled' if base_vocab._c_ext_available else 'Disabled'}")
    except Exception as e:
        print(f"    [ERROR] Failed to load vocabulary: {e}")
        return
    
    # Reconstruct ordered token list and set for O(1) lookup
    print("    - Reconstructing ID mapping...")
    base_tokens = [base_vocab.id_to_token[i] for i in range(len(base_vocab))]
    existing_token_set = set(base_vocab.token_to_id.keys())
    
    # 2. Train on Code Samples
    print(f"\n[2] Training on comprehensive code samples...")
    print("    Languages: Python, JavaScript, TypeScript, Java, C++, Rust, Go")
    print()
    
    start_time = time.time()
    
    # Train vocabulary on code data
    code_tokens_raw = train_vocabulary(
        yield_all_code_samples(),
        target_size=30000,  # Extract up to 30k code tokens
        min_frequency=2,    # Require at least 2 occurrences
        progress_callback=progress_callback
    )
    
    training_time = time.time() - start_time
    print(f"\n    - Extracted {len(code_tokens_raw):,} candidate tokens in {training_time:.1f}s")
    
    # 3. Merge Tokens (Append-Only, ID-Stable)
    print(f"\n[3] Merging new tokens (append-only)...")
    
    new_tokens = []
    skipped = 0
    
    for token in code_tokens_raw:
        if token not in existing_token_set:
            new_tokens.append(token)
            existing_token_set.add(token)  # Prevent duplicates within batch
        else:
            skipped += 1
    
    print(f"    - Existing tokens skipped: {skipped:,}")
    print(f"    - NEW tokens to add:       {len(new_tokens):,}")
    
    # Show sample of new tokens
    if new_tokens:
        print(f"\n    Sample new tokens (first 30):")
        for i, token in enumerate(new_tokens[:30]):
            display = repr(token) if len(token) < 25 else repr(token[:22] + "...")
            print(f"      [{i:2d}] {display}")
    
    # 4. Create Final Vocabulary
    print(f"\n[4] Creating final vocabulary...")
    final_token_list = base_tokens + new_tokens
    
    print(f"    - Base vocabulary:  {len(base_tokens):,}")
    print(f"    - New code tokens:  {len(new_tokens):,}")
    print(f"    - Total vocabulary: {len(final_token_list):,}")
    
    final_vocab = CrayonVocab(final_token_list)
    print(f"    - C-Extension: {'Enabled' if final_vocab._c_ext_available else 'Disabled'}")
    
    # 5. Save Updated Vocabulary
    print(f"\n[5] Saving to {EXISTING_VOCAB_PATH}...")
    final_vocab.save(str(EXISTING_VOCAB_PATH), format="json")
    final_vocab.save("trained_vocab.txt", format="txt")
    print(f"    [DONE] Vocabulary updated successfully!")
    
    # 6. Verification
    print("\n" + "=" * 60)
    print("Verification Tests")
    print("=" * 60)
    
    test_cases = [
        ("Python", "def fibonacci(n: int) -> int:\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
        ("JavaScript", "const fetchData = async (url) => { const res = await fetch(url); return res.json(); }"),
        ("TypeScript", "interface User { id: number; name: string; email: string; }"),
        ("Java", "public static void main(String[] args) { System.out.println(\"Hello World\"); }"),
        ("C++", "#include <iostream>\nint main() { std::cout << \"Hello\" << std::endl; return 0; }"),
        ("Rust", "fn main() { let x: i32 = 42; println!(\"Value: {}\", x); }"),
        ("Go", "func main() { fmt.Println(\"Hello, World!\") }"),
        ("NumPy", "import numpy as np\ndf = pd.DataFrame(data)"),
    ]
    
    for lang, test_str in test_cases:
        tokens = final_vocab.tokenize(test_str)
        decoded = final_vocab.decode(tokens)
        
        # Truncate display for long strings
        display_input = test_str[:50] + "..." if len(test_str) > 50 else test_str
        display_input = display_input.replace('\n', '\\n')
        
        match = '[OK]' if decoded == test_str else '[FAIL]'
        print(f"\n[{lang}]")
        print(f"  Input:   '{display_input}'")
        print(f"  Tokens:  {len(tokens)} tokens | Match: {match}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Original vocabulary: {base_size:,} tokens")
    print(f"  Final vocabulary:    {len(final_vocab):,} tokens")
    print(f"  New tokens added:    {len(new_tokens):,}")
    print(f"  Training time:       {training_time:.1f}s")
    print(f"  Output file:         {EXISTING_VOCAB_PATH}")
    print()


if __name__ == "__main__":
    main()
