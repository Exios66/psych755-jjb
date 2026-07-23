# GetStarted.md

## A one-hour setup session for your group

Do this together, in one room, at one time, with everyone’s laptop open. It takes about an hour (but could take more). Do not try to do it over text message across three days. The steps depend on each other, and doing them out of order is how groups break their repository in week one.

By the end of this session your group will have a repository, everyone will have made their first commit, and everyone will have opened and merged their first pull request. That’s the whole goal.

---

## Part 0: Before you start

### Pick your letters

Sit down and decide, out loud, right now:

- **Student A** This student will be the one to creates the repository. Later as members contribute this student will be the one who also merges everyone’s work. Hint: Pick someone whose laptop is charged and whose GitHub account already exists.
- **Student B** This student will be a collaborator in the repository. Student B will review Student A's pull requests.
- **Student C**
- **Student D**
- **Student E** (Skip if your group has four people).

Write the letters down on paper with names to help you remember who is who. The rest of this guide refers to you by letter.

- **Students A-E:** Jack J. Burleson

### Everyone: Get an account (if you don't have one)

If you don't have a GitHub account, make one now at <https://github.com/signup>. Use a username you'd be glad to show an employer.

### Everyone: install two programs (if you haven't already)

You need `git` (the program that tracks changes) and `gh` (GitHub's command-line tool). You will use `gh` at least once, in the log-in step below, to connect your laptop to your GitHub account. Every other interaction with GitHub in this guide is either a `git` command or a click on the GitHub website.

**On macOS**, open the Terminal app and run:

```
brew install git gh
```

If that says `brew: command not found`, install Homebrew first from <https://brew.sh>, then run the command again.

**On Windows**, install [Git for Windows](https://git-scm.com/download/win) and [GitHub CLI](https://cli.github.com). Then open **Git Bash** (not Command Prompt, not PowerShell) for everything below.

**Check that both worked.** Everyone run these two commands:

```
git --version
gh --version
```

Each should print a version number. If either says "command not found," stop and fix it before going on. Get help from others in your group if you need it. Do not let one person get left behind here.

### Everyone: tell git who you are (if you haven't already)

Git stamps your name and email on every commit you make. If you skip this, your commits will be labeled with something useless like `root@laptop.local`, and we will not be able to tell that the work is yours.

Run these three, with **your own** name and email:

```
git config --global user.name "Beth Chen"
git config --global user.email "bchen42@wisc.edu"
git config --global init.defaultBranch main
```

**Important:** the email must be one that's on your GitHub account. Check what's on your account at <https://github.com/settings/emails> (hint: find the anonymized email address GitHub provides).

### Everyone: log in (if you haven't already)

```
gh auth login
```

It will ask you several questions. Answer:

- **What account?** → GitHub.com
- **Preferred protocol?** → HTTPS
- **Authenticate Git with your GitHub credentials?** → Yes
- **How would you like to authenticate?** → Login with a web browser

In the terminal `gh auth` will an eight-character code. Copy it, paste the code into the browser window that opens, and approve. Come back to the terminal to proceed.

Everyone do this now. Wait until all five of you are done.

---

## Part 1: Student A creates the repository

**Only Student A does this part.** Everyone else, watch A's screen.

1. Create a new repository on GitHub.
2. Set it to **Public**. (Never put passwords, API keys, or private data in this repo.)
3. Click **Create repository**.

### Student A: Add Student B as a colaborator

Still on the new repository page:

1. Click **Settings** (top of the page).
2. Click **Collaborators** in the left sidebar.
3. Click **Add people**.
4. Type Student B's GitHub username, and add them.

**B** check your email now. GitHub sent you an invitation. Click the link and accept it. You cannot do anything else in this guide until you accept.

### Student C: Clone the repository and add new files

1. Fork the repository to your own GitHub account
2. Clone the forked repository to your computer
3. Copy `contributions.md` to your cloned repository
4. Copy `README.md` to your cloned repository
5. Revise away templated placeholders in both files.
6. Commit and push your changes to your forked repository.
7. Open a pull request for Student A to review and pull into the main repository.

### Student A: Review and merge Student C's pull request

1. Go to the pull request page
2. Review the changes
3. Merge the pull request

---

## Part 2: Discuss and plan your project structure

What folder will you create to store original data files?
What folder will you create to store cleaned data files?
How will you organize each group member's memos? (Hint: Create a subfolder for each member)
What other organizational structures will you use?

---

## Part 3: Implement your organizational decisoins

### Student D: Clone the repository and implement the organizational structure discussed in Part 2

1. Fork the repository to your own GitHub account
2. Clone the forked repository to your computer
3. Implement the organizational structure discussed in Part 2 (Create folders)
4. Commit and push your changes to your forked repository
5. Open a pull request for Student A to review and pull into the main repository

### Student B: Review and approve Student D's pull request

1. Go to the pull request page
2. Review the changes
3. Merge the pull request

---

## Part 3: Practice the loop

Each student in turn adds their own row to the members table in `README.md`. **Recommended to proceed one person at a time, in letter order.**

> ### Going in order helps learn the process
>
> **Wait your turn, and pull before you start.**
>
> You are all editing the same (or similar) lines of the same file. If two of you work at the same time, you may experience "merge conflicts" (git will not know whose version is right), which is fixable, but not what you want in your first hour.
>
> Student A goes completely, start to finish. **Then** B starts. Then C. Then D. Then E. Of course also execute a pull request and merge between each student's efforts. To avoid the conflicts, nobody starts their turn until the person before them is merged and Student A (or Student B) says "go."

### Your turn: the six steps

Start this practice loop with Student B. When it's your turn, run these. The example is Student B; use your own name.

**Step 0 - Pull the latest version to your own forked repository**

Go to your forked repository on GitHub and pull the latest changes to ensure you have the most recent version.

**Step 1 — Clone the latest version from your forked repository.**

```
git fetch
git pull
```

The commands `git fetch` and `git pull` download whatever your teammates have already merged. Skipping this is the single most common way to create a mess.

**Step 2 — Edit the file.**

Open `README.md` in whatever editor you like. Find the members table near the top. It looks like this:

```
| Name | GitHub username |
|---|---|
| | |
| | |
```

Fill in **one** empty row (yours) with your real name and your GitHub username:

```
| Name | GitHub username |
|---|---|
| Beth Chen | bchen42 |
| | |
```

Leave the other rows alone. Save the file.

**Step 4 — Save your change to git.**

```
git add README.md
git commit -m "Add Beth Chen to members table"
```

`git add` says "this is the file I changed." `git commit` says "save this change, with this description." The description in quotes is a *commit message.* A reminder is that commit messages let you quickly review your git commit history to understand what edits were in which commits.

**Step 5 — Push your work to your forked repository on GitHub.**

```
git push
```

Your branch now exists on GitHub. Your change still isn't in the main version. Currently, it's sitting in a side room, waiting to be let in.

**Step 6 — Open a pull request on the GitHub website.**

Go to your repository page on github.com (your fork) and refresh it. Find the pull reqeusts section, click new pull request, follow the on-screen instructions to create a pull request that will pull and merge your forked changes into your group's main repository.

Remember that, a pull request is a formal request to merge your work into the main version. Once created, the pull request has its own URL in your browser's address bar — copy it and share it with Student A so they can review.

Your turn is over. Now A takes over. (Also note that Student B could also now respond to your pull request if they your your group wanted B to so.)

### Student A: review and merge

Every time a teammate submits a pull request, then Student A (or Student B) will need to do as follows.

Review the pull requests on the GitHub Website. **Look at what actually changed using the "Files changed" tab.**

Green lines with `+` were added. Red lines with `-` were removed. Actually look. You are checking two things: did they add their own row, and did they leave everyone else's alone? This takes five seconds and it is the entire point of a pull request.

**Let those changes in:**

On the GitHub website, click the "Merge pull request" button.

Now say "go" to the next student, out loud. They will pull your merge down in their step 1 and build on top of it.

---

## Part 4: Student A's turn

Student A goes last, and Student A does not merge their own work. Somebody else has to read it (that's true for the rest of the semester too).

**Student A:** Do the six *look practice* steps in Part 3 exactly like everyone else, with your own name.

**Student B** ou merge Student A's pull request. Again using GitHub website, click the "Merge pull request" button.

---

## What today established

- **Student A owns the repository.** A created it and can add or remove people. That's a chore, not a rank. A is not the boss of this project.
- **Student B is a backup collaborator.** B can merge pull requests and make changes to the repository. If Student A becomes unavailable then Student B can step in (again not a rank, it is a chore).
- **Nothing reaches `main` without a pull request.** Not even A's work. Someone other than the author looks at every change before it lands.
- **Every one in each group has a working setup.** Git is installed, you're authenticated, your commits carry your name, and you've been through the whole loop once on a change that didn't matter.
