"""
System Audit Script - Comprehensive Health Check
PATTERN: Automated assessment for tech debt and issues
"""
import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class SystemAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues = []
        self.warnings = []
        self.info = []
        
    def run_full_audit(self) -> Dict:
        """Run comprehensive system audit"""
        print("🔍 Starting System Audit...\n")
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "python_version": sys.version,
            "project_root": str(self.project_root),
            "checks": {}
        }
        
        # 1. Dependency Check
        print("1️⃣ Checking Dependencies...")
        results["checks"]["dependencies"] = self.check_dependencies()
        
        # 2. Import Check
        print("2️⃣ Checking Module Imports...")
        results["checks"]["imports"] = self.check_imports()
        
        # 3. Configuration Check
        print("3️⃣ Checking Configuration...")
        results["checks"]["configuration"] = self.check_configuration()
        
        # 4. Code Quality Metrics
        print("4️⃣ Analyzing Code Quality...")
        results["checks"]["code_quality"] = self.analyze_code_quality()
        
        # 5. Security Check
        print("5️⃣ Checking Security...")
        results["checks"]["security"] = self.check_security()
        
        # 6. Database Check
        print("6️⃣ Checking Database Setup...")
        results["checks"]["database"] = self.check_database()
        
        # 7. File Structure
        print("7️⃣ Validating File Structure...")
        results["checks"]["structure"] = self.check_file_structure()
        
        # Summary
        results["summary"] = {
            "issues": self.issues,
            "warnings": self.warnings,
            "info": self.info,
            "health_score": self.calculate_health_score()
        }
        
        return results
    
    def check_dependencies(self) -> Dict:
        """Check if all required dependencies are installable"""
        result = {"status": "unknown", "details": []}
        
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            self.issues.append("requirements.txt not found")
            return {"status": "error", "details": ["requirements.txt missing"]}
        
        with open(req_file) as f:
            requirements = f.readlines()
        
        # Check for version conflicts
        packages = {}
        for req in requirements:
            req = req.strip()
            if req and not req.startswith("#"):
                if "==" in req:
                    name, version = req.split("==")
                    if name in packages:
                        self.warnings.append(f"Duplicate package: {name}")
                    packages[name] = version
                    result["details"].append(f"✓ {name}=={version}")
        
        result["status"] = "ok"
        result["package_count"] = len(packages)
        return result
    
    def check_imports(self) -> Dict:
        """Test if all modules can be imported"""
        result = {"status": "unknown", "importable": [], "failed": []}
        
        # Add project root to path
        sys.path.insert(0, str(self.project_root))
        
        modules_to_test = [
            "app.config",
            "app.database",
            "app.state_manager",
            "app.conversation_handler",
            "app.audio_pipeline",
            "app.models",
            "app.twilio_handler",
            "app.main"
        ]
        
        for module in modules_to_test:
            try:
                # Try to import
                exec(f"import {module}")
                result["importable"].append(module)
            except ImportError as e:
                result["failed"].append({"module": module, "error": str(e)})
                self.issues.append(f"Cannot import {module}: {e}")
            except Exception as e:
                result["failed"].append({"module": module, "error": str(e)})
                self.warnings.append(f"Error importing {module}: {e}")
        
        result["status"] = "ok" if not result["failed"] else "error"
        return result
    
    def check_configuration(self) -> Dict:
        """Check configuration and environment variables"""
        result = {"status": "unknown", "env_vars": {}, "missing": []}
        
        # Check for .env.example
        env_example = self.project_root / ".env.example"
        if not env_example.exists():
            self.warnings.append(".env.example not found")
            return result
        
        # Parse required env vars from example
        with open(env_example) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key = line.split("=")[0]
                    # Check if it's set in environment
                    value = os.environ.get(key)
                    if value:
                        result["env_vars"][key] = "SET"
                    else:
                        result["missing"].append(key)
        
        # Check critical vars
        critical = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
        for var in critical:
            if var in result["missing"]:
                self.issues.append(f"Critical env var missing: {var}")
        
        result["status"] = "ok" if not result["missing"] else "warning"
        return result
    
    def analyze_code_quality(self) -> Dict:
        """Analyze code quality metrics"""
        result = {"status": "unknown", "metrics": {}}
        
        # Count lines of code
        py_files = list((self.project_root / "app").glob("*.py"))
        total_lines = 0
        total_functions = 0
        total_classes = 0
        
        for py_file in py_files:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                total_lines += len(lines)
                total_functions += content.count('def ')
                total_classes += content.count('class ')
        
        result["metrics"] = {
            "files": len(py_files),
            "total_lines": total_lines,
            "functions": total_functions,
            "classes": total_classes,
            "avg_lines_per_file": total_lines // len(py_files) if py_files else 0
        }
        
        # Check for common issues
        issues_found = []
        
        for py_file in py_files:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for TODOs
                if 'TODO' in content:
                    issues_found.append(f"TODO found in {py_file.name}")
                    self.info.append(f"TODO in {py_file.name}")
                
                # Check for print statements (should use logging)
                if '\nprint(' in content:
                    issues_found.append(f"print() statements in {py_file.name}")
                    self.warnings.append(f"Using print() instead of logging in {py_file.name}")
                
                # Check for bare except
                if 'except:' in content:
                    issues_found.append(f"Bare except in {py_file.name}")
                    self.warnings.append(f"Bare except clause in {py_file.name}")
        
        result["issues"] = issues_found
        result["status"] = "ok" if not issues_found else "warning"
        return result
    
    def check_security(self) -> Dict:
        """Basic security checks"""
        result = {"status": "unknown", "vulnerabilities": []}
        
        # Check for hardcoded secrets
        patterns_to_check = [
            ("API_KEY", r'["\'].*api[_-]?key.*["\']\s*=\s*["\'][^"\']+["\']'),
            ("SECRET", r'["\'].*secret.*["\']\s*=\s*["\'][^"\']+["\']'),
            ("PASSWORD", r'["\'].*password.*["\']\s*=\s*["\'][^"\']+["\']'),
        ]
        
        py_files = list((self.project_root / "app").glob("*.py"))
        
        for py_file in py_files:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                
                for pattern_name, pattern in patterns_to_check:
                    # Simplified check - just look for keywords
                    if pattern_name.lower() in content:
                        # Check if it's not from environment
                        if 'os.environ' not in content and 'settings.' not in content:
                            result["vulnerabilities"].append({
                                "file": py_file.name,
                                "type": pattern_name,
                                "severity": "high"
                            })
                            self.issues.append(f"Potential hardcoded {pattern_name} in {py_file.name}")
        
        # Check for SQL injection vulnerabilities
        for py_file in py_files:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Look for string formatting in SQL
                if 'execute(' in content and ('%' in content or '.format(' in content):
                    self.warnings.append(f"Potential SQL injection risk in {py_file.name}")
                    result["vulnerabilities"].append({
                        "file": py_file.name,
                        "type": "SQL_INJECTION",
                        "severity": "medium"
                    })
        
        result["status"] = "ok" if not result["vulnerabilities"] else "warning"
        return result
    
    def check_database(self) -> Dict:
        """Check database configuration"""
        result = {"status": "unknown", "details": {}}
        
        # Check if database file exists or can be created
        db_path = self.project_root / "learning_captures.db"
        result["details"]["db_exists"] = db_path.exists()
        result["details"]["db_path"] = str(db_path)
        
        # Check for migrations or schema management
        if not (self.project_root / "migrations").exists():
            self.info.append("No migrations directory found - consider adding schema versioning")
        
        result["status"] = "ok"
        return result
    
    def check_file_structure(self) -> Dict:
        """Validate project file structure"""
        result = {"status": "unknown", "structure": {}}
        
        expected_structure = {
            "app/": ["__init__.py", "main.py", "config.py", "database.py"],
            "static/": ["index.html", "manifest.json", "sw.js"],
            "scripts/": [],
            "tests/": []
        }
        
        for dir_path, expected_files in expected_structure.items():
            dir_full = self.project_root / dir_path.rstrip('/')
            result["structure"][dir_path] = {
                "exists": dir_full.exists(),
                "files": []
            }
            
            if dir_full.exists():
                for expected_file in expected_files:
                    file_path = dir_full / expected_file
                    if file_path.exists():
                        result["structure"][dir_path]["files"].append(f"✓ {expected_file}")
                    else:
                        result["structure"][dir_path]["files"].append(f"✗ {expected_file}")
                        if dir_path == "app/":
                            self.issues.append(f"Missing critical file: {dir_path}{expected_file}")
                        else:
                            self.warnings.append(f"Missing file: {dir_path}{expected_file}")
            else:
                if dir_path == "tests/":
                    self.warnings.append("No tests directory - testing not implemented")
                elif dir_path != "scripts/":
                    self.issues.append(f"Missing directory: {dir_path}")
        
        result["status"] = "ok" if not any(i for i in self.issues if "Missing" in i) else "warning"
        return result
    
    def calculate_health_score(self) -> int:
        """Calculate overall health score (0-100)"""
        score = 100
        
        # Deduct points for issues
        score -= len(self.issues) * 10
        score -= len(self.warnings) * 3
        
        # Ensure score is between 0 and 100
        return max(0, min(100, score))
    
    def print_summary(self, results: Dict):
        """Print formatted summary"""
        print("\n" + "="*60)
        print("📊 AUDIT SUMMARY")
        print("="*60)
        
        health_score = results["summary"]["health_score"]
        
        # Health score with color coding
        if health_score >= 80:
            status = "🟢 HEALTHY"
        elif health_score >= 60:
            status = "🟡 NEEDS ATTENTION"
        else:
            status = "🔴 CRITICAL ISSUES"
        
        print(f"\nHealth Score: {health_score}/100 - {status}")
        
        # Issues
        if results["summary"]["issues"]:
            print(f"\n❌ Critical Issues ({len(results['summary']['issues'])}):")
            for issue in results["summary"]["issues"]:
                print(f"   • {issue}")
        
        # Warnings
        if results["summary"]["warnings"]:
            print(f"\n⚠️  Warnings ({len(results['summary']['warnings'])}):")
            for warning in results["summary"]["warnings"][:5]:  # Show first 5
                print(f"   • {warning}")
            if len(results["summary"]["warnings"]) > 5:
                print(f"   ... and {len(results['summary']['warnings']) - 5} more")
        
        # Info
        if results["summary"]["info"]:
            print(f"\nℹ️  Information ({len(results['summary']['info'])}):")
            for info in results["summary"]["info"][:3]:  # Show first 3
                print(f"   • {info}")
        
        # Recommendations
        print("\n📋 RECOMMENDATIONS:")
        self.print_recommendations(results)
        
        print("\n" + "="*60)
    
    def print_recommendations(self, results: Dict):
        """Generate and print recommendations based on audit results"""
        recommendations = []
        
        # Check imports
        if results["checks"].get("imports", {}).get("failed"):
            recommendations.append("1. Fix import errors - install missing dependencies")
        
        # Check configuration
        if results["checks"].get("configuration", {}).get("missing"):
            recommendations.append("2. Set missing environment variables in .env file")
        
        # Check code quality
        if results["checks"].get("code_quality", {}).get("issues"):
            recommendations.append("3. Address code quality issues (TODOs, logging, exception handling)")
        
        # Check security
        if results["checks"].get("security", {}).get("vulnerabilities"):
            recommendations.append("4. Review and fix security vulnerabilities")
        
        # Check for tests
        if not (self.project_root / "tests").exists():
            recommendations.append("5. Implement unit tests for critical functionality")
        
        # Always recommend
        recommendations.extend([
            "6. Set up continuous integration (CI/CD)",
            "7. Add comprehensive logging throughout the application",
            "8. Implement database migrations for schema management"
        ])
        
        for rec in recommendations[:5]:  # Show top 5
            print(f"   {rec}")

def main():
    """Run the audit"""
    project_root = Path(__file__).parent.parent
    auditor = SystemAuditor(project_root)
    
    results = auditor.run_full_audit()
    
    # Save results to file
    output_file = project_root / "audit_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    auditor.print_summary(results)
    
    print(f"\nFull results saved to: {output_file}")
    
    return results["summary"]["health_score"]

if __name__ == "__main__":
    sys.exit(0 if main() >= 60 else 1)