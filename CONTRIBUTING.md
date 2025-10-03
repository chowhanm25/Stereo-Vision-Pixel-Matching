# Contributing to Stereo Vision Pixel Matching

We welcome contributions to the Stereo Vision Pixel Matching project! This document provides guidelines for contributing to the project.

## 🎯 How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use descriptive titles** and provide detailed information
3. **Include system information** (OS, Python version, OpenCV version)
4. **Provide reproduction steps** for bugs
5. **Add screenshots or error logs** when relevant

### Suggesting Features

1. **Open a feature request** issue
2. **Describe the use case** and expected behavior
3. **Explain the implementation approach** if you have ideas
4. **Consider alternatives** and mention trade-offs

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit with clear messages**
7. **Push and create a pull request**

## 🔧 Development Setup

### Prerequisites
- Python 3.8+
- OpenCV 4.0+
- Git
- Virtual environment tool

### Setup Instructions

1. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Stereo-Vision-Pixel-Matching.git
   cd Stereo-Vision-Pixel-Matching
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run tests:**
   ```bash
   python -m pytest tests/
   ```

## 📝 Coding Standards

### Python Code Style

- Follow **PEP 8** guidelines
- Use **meaningful variable names**
- Write **comprehensive docstrings**
- Keep **line length under 88 characters**
- Use **type hints** where appropriate

### Code Formatting

We use automated tools for code formatting:

```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Lint with flake8
flake8 .
```

### Computer Vision Best Practices

- **Validate input parameters** (image dimensions, data types)
- **Handle edge cases** gracefully
- **Optimize for performance** when possible
- **Add parameter validation** for public methods
- **Use OpenCV conventions** for coordinate systems

## 🧪 Testing Requirements

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_stereo_matcher.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Test Guidelines

- **Write tests** for all new features
- **Include edge cases** and error conditions
- **Use descriptive test names**
- **Mock external dependencies** when appropriate
- **Test both success and failure paths**

### Test Types

1. **Unit Tests**: Test individual methods and functions
2. **Integration Tests**: Test component interactions
3. **Visual Tests**: Compare output images (when applicable)
4. **Performance Tests**: Benchmark critical algorithms

## 📚 Documentation

### Documentation Requirements

- **Update README.md** for new features
- **Add docstrings** to all public methods
- **Include code examples** in documentation
- **Update API reference** for interface changes
- **Write clear commit messages**

### Documentation Style

- Use **clear, concise language**
- Include **code examples**
- Add **mathematical explanations** for algorithms
- **Link to relevant papers** or resources

## 🚀 Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
2. **Update documentation**
3. **Check code formatting**
4. **Write descriptive commit messages**
5. **Rebase on latest main branch**

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Tested with different image types and sizes
- [ ] Verified backward compatibility

## Algorithm Changes
- [ ] Algorithm modifications are well-documented
- [ ] Performance impact is assessed
- [ ] Accuracy improvements are quantified

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly complex algorithms
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or feature works
- [ ] New and existing unit tests pass locally
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Algorithm validation** for computer vision changes
4. **Performance testing** for critical path modifications
5. **Documentation review** for completeness
6. **Final approval** and merge

## 🎨 Computer Vision Guidelines

### Algorithm Implementation

- **Document mathematical foundations** in code comments
- **Cite relevant papers** in docstrings
- **Validate against known benchmarks** when possible
- **Handle different image formats** (grayscale, color, different bit depths)
- **Consider numerical stability** in computations

### Performance Considerations

- **Profile critical sections** of code
- **Use vectorized operations** when possible
- **Consider memory usage** for large images
- **Optimize for common use cases**
- **Document performance characteristics**

### Error Handling

- **Validate input images** (not None, correct dimensions)
- **Check parameter ranges** (positive window sizes, valid search ranges)
- **Handle edge cases** gracefully (empty matches, degenerate cases)
- **Provide meaningful error messages**

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: Contact maintainers directly

### Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)
- [Multiple View Geometry](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- [Python Style Guide](https://pep8.org/)

## 🏆 Recognition

Contributors will be:
- **Listed** in project documentation
- **Credited** in release notes
- **Acknowledged** in academic papers (if applicable)
- **Invited** to join the maintainer team (for significant contributions)

## 📝 Priority Areas for Contribution

### High Priority
- **Performance optimization** of ZNCC computation
- **Additional correlation methods** (SAD, SSD, NCC)
- **Robust error handling** and edge case management
- **Comprehensive test suite** development

### Medium Priority
- **Dense stereo matching** implementation
- **Sub-pixel accuracy** improvements
- **Additional feature detectors** (ORB, AKAZE)
- **Interactive parameter tuning** interface

### Future Enhancements
- **Real-time video processing**
- **GPU acceleration** with OpenCV's GPU modules
- **3D reconstruction** visualization
- **Mobile/web deployment** options

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

---

**Thank you for contributing to the Stereo Vision Pixel Matching project!** 🎉

Your contributions help advance computer vision education and research, making stereo vision concepts more accessible to students and researchers worldwide.