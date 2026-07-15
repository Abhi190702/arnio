## Summary

The project README has grown into a combined landing page, tutorial, API
reference, benchmark report, architecture guide, troubleshooting manual,
contributor handbook, and release guide.

At the current `main` revision it contains approximately:

- 2,163 lines
- 9,000 words
- 90 code blocks
- 40% code by line count

This makes the first-time project experience difficult to scan and creates
duplicate documentation that can drift away from the website and dedicated
repository guides.

The project also needs a maintainable way to recognize both implementation
contributors and people who report accepted issues without placing hundreds
of profiles directly in the README.

## Proposed work

### README

- Reduce the README to a focused project gateway.
- Keep installation, one complete quickstart, key capabilities, integrations,
  a compact benchmark summary, documentation links, contributor recognition,
  community links, security, and license information.
- Move detailed guidance to existing canonical documentation instead of
  deleting useful material.

### Documentation structure

- Add or improve a documentation index that directs readers to API,
  architecture, benchmarks, examples, troubleshooting, remote/chunked I/O,
  contributor, GSSoC, roadmap, and security documentation.
- Establish clear ownership so the README, repository Markdown, and website do
  not independently maintain the same long-form content.

### Contributor recognition

- Add a generated `CONTRIBUTORS.md` page.
- Include human authors with a merged pull request.
- Include issue reporters only when the issue has been maintainer-categorized
  with a `type:*` label.
- Exclude bots and issues labeled `duplicate`, `invalid`, or `wontfix`.
- Record whether each person contributed through merged PRs, qualified issue
  reports, or both.

### Website

- Add a dedicated contributor gallery with searchable/filterable profiles.
- Add contributor navigation and cross-links from community/contributing pages.
- Keep the README contributor section compact and link to the complete gallery.
- Preserve responsive behavior, accessibility, theme support, and existing
  website conventions.

### Automation

- Add a deterministic contributor-data generator.
- Generate both `CONTRIBUTORS.md` and website JSON from the same data model.
- Add tests that validate the generated files and eligibility rules.

## Acceptance criteria

- [ ] README is substantially shorter and retains a clear first-run path.
- [ ] Removed README details remain discoverable in canonical documentation.
- [ ] Documentation links are valid and organized by audience.
- [ ] `CONTRIBUTORS.md` is generated deterministically.
- [ ] Contributor eligibility excludes bots, duplicate, invalid, and wontfix
      issue reports.
- [ ] Website has a responsive, accessible contributor gallery.
- [ ] Contributor gallery supports useful categories such as all, merged PRs,
      issue reporters, and both.
- [ ] Website navigation and metadata include the new page.
- [ ] Repository and website tests pass.
- [ ] The change does not manually duplicate hundreds of contributor profiles
      in the README.

## Impact

This improves project onboarding, documentation maintainability, contributor
recognition, website discoverability, and the professional presentation of
Arnio without losing the detailed technical material already created.
