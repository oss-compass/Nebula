#!/usr/bin/env python3

import sys
import ast
import os
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("�?Python版本兼容")
        return True
    else:
        print("�?Python版本过低，需要Python 3.8+")
        return False

def check_syntax_compatibility(file_path):
    """检查单个文件的语法兼容�?""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试解析AST
        ast.parse(content)
        return True, "语法正确"
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def check_imports_compatibility(file_path):
    """检查导入兼容�?""
    incompatible_imports = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['typing_extensions']:
                        # typing_extensions在Python 3.8中可�?
                        continue
            elif isinstance(node, ast.ImportFrom):
                if node.module == 'typing':
                    for alias in node.names:
                        # 检查Python 3.8不支持的typing特�?
                        if alias.name in ['Literal', 'TypedDict', 'Final']:
                            # 这些在Python 3.8中需要typing_extensions
                            if 'typing_extensions' not in content:
                                incompatible_imports.append(f"{alias.name} 需�?typing_extensions")
    
    except Exception as e:
        return [f"解析错误: {e}"]
    
    return incompatible_imports

def check_python38_features():
    """检查是否使用了Python 3.8不支持的特�?""
    issues = []
    
    # 检查walrus operator (:=)
    if ':=' in open(__file__).read():
        issues.append("使用了海象运算符 (:=)，需要Python 3.8+")
    
    # 检查positional-only parameters
    # 这个需要更复杂的AST分析
    
    return issues

def main():
    """主函�?""
    print("=" * 50)
    print("Python 3.8兼容性检�?)
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        return
    
    print("\n检查search模块文件...")
    
    search_dir = Path(__file__).parent
    python_files = list(search_dir.glob("*.py"))
    
    all_good = True
    
    for py_file in python_files:
        if py_file.name == __file__.split('/')[-1]:  # 跳过当前文件
            continue
            
        print(f"\n检查文�? {py_file.name}")
        
        # 检查语�?
        syntax_ok, syntax_msg = check_syntax_compatibility(py_file)
        if syntax_ok:
            print(f"  �?语法: {syntax_msg}")
        else:
            print(f"  �?语法: {syntax_msg}")
            all_good = False
        
        # 检查导�?
        import_issues = check_imports_compatibility(py_file)
        if not import_issues:
            print(f"  �?导入: 兼容")
        else:
            print(f"  �?导入问题:")
            for issue in import_issues:
                print(f"    - {issue}")
            all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("�?所有文件都与Python 3.8兼容�?)
    else:
        print("�?发现兼容性问题，请修复后重新检�?)
    print("=" * 50)
    
    # 提供兼容性建�?
    print("\nPython 3.8兼容性建�?")
    print("1. 使用 typing.List, typing.Dict 而不�?list[], dict[]")
    print("2. 避免使用海象运算�?(:=)")
    print("3. 避免使用 match-case 语句")
    print("4. 避免使用字典合并运算�?(|)")
    print("5. 如需使用新特性，考虑使用 typing_extensions")

if __name__ == "__main__":
    main()
