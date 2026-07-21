"""Repository audit script: imports, definitions, references, duplicates, dead code.
Outputs audit_output.json."""
import ast
import json
import os
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
EXCLUDE_DIRS = {'.venv', 'test_torch', '__pycache__', '.git'}


def iter_py_files():
    for p in ROOT.rglob('*.py'):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        yield p


def module_name_from_path(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = rel.with_suffix('').parts
    # ignore __init__ at top for module name? e.g. src/__init__.py -> src
    return '.'.join(parts)


def extract_imports(node):
    imports = []
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for a in n.names:
                imports.append({
                    'type': 'import',
                    'module': a.name,
                    'as': a.asname,
                    'names': [{'name': a.name, 'as': a.asname}]
                })
        elif isinstance(n, ast.ImportFrom):
            module = n.module or ''
            level = n.level
            names = [{'name': a.name, 'as': a.asname} for a in n.names]
            imports.append({'type': 'from', 'module': module, 'level': level, 'names': names})
    return imports


def resolve_imported_module(imp, current_module: str):
    """Return a list of candidate modules referenced by an import statement."""
    if imp['type'] == 'import':
        return [imp['module']]
    # from import
    base = current_module.split('.') if current_module else []
    if imp['level']:
        # relative
        if imp['level'] > len(base):
            return []
        prefix = '.'.join(base[:len(base) - imp['level']]) if imp['level'] <= len(base) else ''
        mod = (prefix + '.' + imp['module']).strip('.') if imp['module'] else prefix
        if not mod:
            mod = '.'.join(base[:len(base) - imp['level']])
        return [mod] if mod else []
    else:
        return [imp['module']] if imp['module'] else []


def resolve_imported_names(imp, current_module: str):
    """Return mapping of alias -> imported full module name (or object)."""
    mod = resolve_imported_module(imp, current_module)
    names = {}
    if imp['type'] == 'import':
        # import a.b as c -> local name c, used as c or a.b
        for alias in imp['names']:
            local = alias['as'] if alias['as'] else alias['name'].split('.')[0]
            names[local] = alias['name']
    else:
        for n in imp['names']:
            local = n['as'] if n['as'] else n['name']
            names[local] = (imp['module'] or '', n['name'])
    return names


def extract_definitions(node):
    classes = []
    functions = []
    assigns = []
    for n in node.body:
        if isinstance(n, ast.ClassDef):
            classes.append({'name': n.name, 'bases': [ast.unparse(b) for b in n.bases], 'lineno': n.lineno})
        elif isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef):
            functions.append({'name': n.name, 'lineno': n.lineno})
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigns.append(t.id)
        elif isinstance(n, ast.AnnAssign):
            if isinstance(n.target, ast.Name):
                assigns.append(n.target.id)
    return classes, functions, assigns


def extract_references(node):
    """Return set of names and attributes referenced."""
    names = set()
    attrs = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            # collect full dotted prefix
            attrs.add(ast.unparse(n))
            # also first component
            first = n.value
            while isinstance(first, ast.Attribute):
                first = first.value
            if isinstance(first, ast.Name):
                names.add(first.id)
    return names, attrs


def has_main_guard(node):
    for n in ast.walk(node):
        if isinstance(n, ast.If):
            if isinstance(n.test, ast.Compare):
                left = n.test.left
                if isinstance(left, ast.Name) and left.id == '__name__':
                    for comp in n.test.comparators:
                        if isinstance(comp, ast.Constant) and comp.value == '__main__':
                            return True
    return False


def find_debug_markers(source: str, classes, functions):
    markers = []
    # print calls
    for match in re.finditer(r'\bprint\s*\(', source):
        line = source[:match.start()].count('\n') + 1
        markers.append({'type': 'print', 'lineno': line, 'snippet': source[match.start():match.start()+60].split('\n')[0]})
    # trace markers
    for label in ['BOOTSTRAP-TRACE', 'META TRACE', 'DEBUG TRACE', 'TEMP LOG', 'TODO DEBUG']:
        for match in re.finditer(re.escape(label), source, re.IGNORECASE):
            line = source[:match.start()].count('\n') + 1
            markers.append({'type': label, 'lineno': line})
    # overly long function/classes with prints
    return markers


def category_from_path(rel):
    parts = Path(rel).parts
    if 'tests' in parts:
        return 'tests'
    if 'benchmarks' in parts:
        return 'benchmarks'
    if 'scripts' in parts:
        return 'scripts'
    if rel.startswith('src'):
        return 'src'
    return 'root'


def main():
    files = {}
    module_to_file = {}
    for path in iter_py_files():
        rel = str(path.relative_to(ROOT))
        mod = module_name_from_path(path)
        module_to_file[mod] = str(rel)
        try:
            source = path.read_text(encoding='utf-8', errors='replace')
            node = ast.parse(source, filename=str(rel))
        except SyntaxError as e:
            files[rel] = {'error': str(e)}
            continue
        imports = extract_imports(node)
        classes, functions, assigns = extract_definitions(node)
        names, attrs = extract_references(node)
        markers = find_debug_markers(source, classes, functions)
        files[rel] = {
            'module': mod,
            'category': category_from_path(rel),
            'imports': imports,
            'definitions': {
                'classes': classes,
                'functions': functions,
                'variables': assigns,
            },
            'references': sorted(names),
            'attrs': sorted(attrs),
            'main_guard': has_main_guard(node),
            'debug_markers': markers,
            'loc': len(source.splitlines()),
        }

    # reverse import map
    imported_by = defaultdict(list)
    for rel, info in files.items():
        if 'imports' not in info:
            continue
        mod = info['module']
        for imp in info['imports']:
            for cand in resolve_imported_module(imp, mod):
                # map to file if exact module matches or submodule
                if cand in module_to_file:
                    imported_by[module_to_file[cand]].append(rel)
                else:
                    # try importing package (directory __init__)
                    parts = cand.split('.')
                    for i in range(len(parts), 0, -1):
                        pkg = '.'.join(parts[:i])
                        if pkg in module_to_file:
                            imported_by[module_to_file[pkg]].append(rel)
                            break
                    # also consider if cand is a package with __init__ inside a file (e.g. src as namespace)
                    if cand.startswith('src.') and cand[4:] in module_to_file:
                        imported_by[module_to_file[cand[4:]]].append(rel)

    # duplicates
    class_counts = defaultdict(list)
    func_counts = defaultdict(list)
    for rel, info in files.items():
        if 'definitions' not in info:
            continue
        for c in info['definitions']['classes']:
            class_counts[c['name']].append(rel)
        for f in info['definitions']['functions']:
            func_counts[f['name']].append(rel)

    # unused imports per file
    for rel, info in files.items():
        if 'imports' not in info:
            continue
        refs = set(info['references'])
        unused = []
        for imp in info['imports']:
            names = resolve_imported_names(imp, info['module'])
            for local, target in names.items():
                # Heuristic: local name not referenced and not a module alias used as dotted prefix
                if local not in refs and not any(a.startswith(local + '.') for a in info.get('attrs', [])):
                    unused.append({'local': local, 'target': str(target)[:100]})
        info['unused_imports'] = unused

    # dead definitions: defined but not referenced in any file (ignore tests/benchmarks/scripts)
    # For production src files, count how many files reference a class/function name
    all_refs = set()
    for info in files.values():
        if 'references' in info:
            all_refs.update(info['references'])

    report = {
        'file_count': len(files),
        'files': files,
        'imported_by': dict(imported_by),
        'duplicate_classes': {k: v for k, v in class_counts.items() if len(v) > 1},
        'duplicate_functions': {k: v for k, v in func_counts.items() if len(v) > 1},
        'module_to_file': module_to_file,
    }

    out = ROOT / 'audit_output.json'
    out.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
