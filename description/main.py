import argparse
import json
import asyncio
import os
from pathlib import Path
from .cache import description_cache
from .batch_processor import generate, create_complete_data, _create_function_index
from .config import DEFAULT_CONCURRENT, DEFAULT_BATCH_SIZE, PERFORMANCE_NOTES


def main():
    parser = argparse.ArgumentParser(description='Generate API function descriptions based on source code and comments')
    parser.add_argument('input', help='Input JSON file path (extract5.py output format)')
    parser.add_argument('--concurrent', type=int, default=DEFAULT_CONCURRENT, help=f'Concurrency count (default: {DEFAULT_CONCURRENT})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help=f'Batch size for simple functions (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--use-cache', action='store_true', help='Enable caching to speed up processing')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before processing')
    parser.add_argument('--skip-complexity', action='store_true', help='Skip complexity calculation, use moderate for all functions (faster processing)')
    parser.add_argument('--recalculate-metrics', action='store_true', help='Recalculate complexity and importance metrics instead of using extract module output (slower but more accurate)')
    args = parser.parse_args()
    
    # 缓存管理
    if args.clear_cache:
        description_cache.clear()
        print("Cache cleared")
    
    if not args.use_cache:
        description_cache.clear()
        print("Cache disabled")

    # 加载API数据
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            api_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {args.input}")
        return
    except json.JSONDecodeError:
        print(f"Error: {args.input} is not a valid JSON file")
        return

    # 提取函数数据
    functions = []
    
    if 'functions' in api_json and isinstance(api_json['functions'], list):
        functions = api_json['functions']
        print(f"Detected extract5.py output format, found {len(functions)} functions")
    else:
        print("Error: Input file must contain 'functions' field and be in extract5.py output format")
        return

    if not functions:
        print("Error: No function data found")
        return

    print(f"Starting to process {len(functions)} functions...")
    print(f"Concurrent: {args.concurrent}, Batch size: {args.batch_size}")
    if args.skip_complexity:
        print("[WARNING] 跳过复杂度计算模式：所有函数将使用moderate复杂度，处理速度更快")
    
    # 显示性能优化说明
    if args.use_cache or args.skip_complexity or args.recalculate_metrics:
        print("\n" + "="*50)
        print("性能优化说明:")
        print(PERFORMANCE_NOTES)
        if args.skip_complexity:
            print("\n[FAST] 跳过复杂度计算模式说明:")
            print("- 所有函数默认使用moderate复杂度")
            print("- 所有函数都使用批量处理，速度更快")
            print("- 适合快速生成大量函数的描述")
        if args.recalculate_metrics:
            print("\n[RECALCULATE] 重新计算指标模式说明:")
            print("- 忽略extract模块的复杂度信息，重新计算")
            print("- 处理速度较慢，但可能更准确")
            print("- 适合需要重新分析复杂度的场景")
        else:
            print("\n[DEFAULT] 默认模式说明:")
            print("- 优先使用extract模块已计算的复杂度信息")
            print("- 避免重复计算，提高处理速度")
            print("- 保持数据一致性，推荐使用")
        print("="*50 + "\n")
    
    # 生成描述
    final = asyncio.run(generate(functions, args.concurrent, args.batch_size, args.skip_complexity, args.recalculate_metrics))
    
    # 生成输出文件名
    input_filename = Path(args.input).stem
    if "_api_extraction" in input_filename:
        repo_name = input_filename.replace("_api_extraction", "")
    else:
        repo_name = input_filename
    
    # 创建输出目录和子目录
    output_dir = os.path.join("output", "description_output")
    descriptions_dir = os.path.join(output_dir, "descriptions")
    indexes_dir = os.path.join(output_dir, "indexes")
    complete_dir = os.path.join(output_dir, "complete")
    
    os.makedirs(descriptions_dir, exist_ok=True)
    os.makedirs(indexes_dir, exist_ok=True)
    os.makedirs(complete_dir, exist_ok=True)
    
    # 保存主要结果到descriptions子文件夹
    output_file = os.path.join(descriptions_dir, f"{repo_name}_api_description3.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    
    # 生成索引文件到indexes子文件夹
    index_file = os.path.join(indexes_dir, f"{repo_name}_api_description_index.json")
    function_index = _create_function_index(final)
    
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(function_index, f, ensure_ascii=False, indent=2)
    
    # 生成完整的合并文件到complete子文件夹
    complete_file = os.path.join(complete_dir, f"{repo_name}_complete_for_neo4j.json")
    complete_data = create_complete_data(api_json, final, args.skip_complexity)
    
    with open(complete_file, "w", encoding="utf-8") as f:
        json.dump(complete_data, f, ensure_ascii=False, indent=2)
    
    print(f"Complete! Results saved to {output_file}")
    print(f"Function index saved to {index_file}")
    print(f"Complete data for Neo4j saved to {complete_file}")
    
    # 统计结果
    success_count = len([r for r in final.values() if 'error' not in r])
    error_count = len([r for r in final.values() if 'error' in r])
    
    print(f"Successfully processed: {success_count} functions")
    print(f"Failed: {error_count} functions")
    
    # 复杂度统计
    complexity_stats = {"simple": 0, "moderate": 0, "complex": 0, "unknown": 0}
    for result in final.values():
        if 'complexity_info' in result:
            level = result['complexity_info'].get('complexity_level', 'unknown')
            complexity_stats[level] += 1
    
    print(f"\nComplexity distribution:")
    print(f"  Simple: {complexity_stats['simple']}")
    print(f"  Moderate: {complexity_stats['moderate']}")
    print(f"  Complex: {complexity_stats['complex']}")
    print(f"  Unknown: {complexity_stats['unknown']}")
    
    # 显示同名函数统计
    print(f"\nFunction name statistics:")
    name_counts = {}
    for result in final.values():
        if 'function_name' in result:
            name = result['function_name']
            name_counts[name] = name_counts.get(name, 0) + 1
    
    duplicate_names = {name: count for name, count in name_counts.items() if count > 1}
    if duplicate_names:
        print(f"  Functions with duplicate names: {len(duplicate_names)}")
        for name, count in sorted(duplicate_names.items())[:5]:  # 显示前5个
            print(f"    {name}: {count} instances")
        if len(duplicate_names) > 5:
            print(f"    ... and {len(duplicate_names) - 5} more")
    else:
        print(f"  No duplicate function names found")
    
    # 显示缓存统计
    if args.use_cache:
        cache_stats = description_cache.get_stats()
        print(f"\nCache statistics:")
        print(f"  Cache size: {cache_stats['cache_size']} entries")
        print(f"  Cache file: {cache_stats['cache_file']}")
        print(f"  Cache file size: {cache_stats['cache_file_size']} bytes")


if __name__ == '__main__':
    main()
